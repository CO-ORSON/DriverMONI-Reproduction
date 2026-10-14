"""Evidence provenance, context separation and linkage contracts."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from src.preprocessing.brca_driver_evidence import (
    GeneMapper, archive_version, build_status, evidence_row, intogen_evidence,
    read_table, run_driver_evidence, validate_tables,
)


GENE = 'ENSG00000000001'
OTHER = 'ENSG00000000002'


def mapper():
    rows = [{'symbol': symbol, 'status': 'Approved', 'ensembl_gene_id': gene, 'entrez_id': str(i)}
            for i, (symbol, gene) in enumerate([('A', GENE), ('B', OTHER)], 1)]
    return GeneMapper(rows, {'ENST00000000001': {GENE}}, {})


def cohort(name='TCGA_TEST'):
    return {'COHORT': name, 'CANCER': 'BRCA', 'CANCER_NAME': 'Breast carcinoma',
            'SOURCE': 'TCGA' if name.startswith('TCGA') else 'Other cohort',
            'TYPE': 'Primary', 'TREATED': 'Untreated', 'REFERENCE': 'PMID:12345678'}


def raw_call(name='TCGA_TEST', passing=True, flag=None):
    return {'SYMBOL': 'A', 'TRANSCRIPT': 'ENST00000000001', 'COHORT': name,
            'CANCER_TYPE': 'BRCA', 'DRIVER': 'True' if passing else 'False',
            'FILTER': flag or ('PASS' if passing else 'No driver'), 'TIER': '1',
            'QVALUE_COMBINATION': '0.01', 'ALL_METHODS': 'dndscv'}


class EvidenceTests(unittest.TestCase):
    def test_ambiguous_and_conflicting_mapping_never_chooses_first(self):
        rows = [{'symbol': 'A', 'status': 'Approved', 'ensembl_gene_id': gene, 'entrez_id': str(i)}
                for i, gene in enumerate([GENE, OTHER], 1)]
        ambiguous = GeneMapper(rows, {}, {})
        self.assertEqual(ambiguous.resolve('A')['mapping_status'], 'ambiguous_mapping')
        self.assertFalse(ambiguous.resolve('A')['gene_ens'])
        self.assertEqual(mapper().resolve('A', '2')['mapping_status'], 'symbol_ncbi_conflict')
        self.assertEqual(mapper().resolve('B', transcript='ENST00000000001')['mapping_status'], 'transcript_gene_conflict')

    def test_exact_breast_code_pass_filter_and_cohort_scope(self):
        accepted = dict(raw_call())
        rejected = raw_call('OTHER', passing=True, flag='Known artifact')
        lung = dict(raw_call('LUNG'), CANCER_TYPE='LUAD')
        records = intogen_evidence([accepted], [accepted, rejected, lung],
            {'TCGA_TEST': cohort(), 'OTHER': cohort('OTHER')}, mapper(),
            {'source_resource': 'IntOGen', 'source_version': 'fixture'})
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['assertion'], 'BRCA_somatic_driver_call')
        self.assertEqual(records[0]['cohort_overlap'], 'same_project_possible_sample_overlap')
        self.assertEqual(records[1]['assertion'], 'BRCA_unaccepted_driver_call')
        self.assertEqual(records[1]['source_filter'], 'Known artifact')
        self.assertTrue(all(row['eligible_for_somatic_status'] == 0 for row in records))

    def test_compendium_disagreement_is_an_error(self):
        row = raw_call(flag='Warning artifact')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            intogen_evidence([row], [row], {'TCGA_TEST': cohort()}, mapper(), {})

    def test_database_agreement_and_germline_do_not_promote_somatic_status(self):
        records = [evidence_row(mapper().resolve('A'), source_resource=source,
            source_record_id=source, assertion='curated_gene_context', setting='germline',
            cancer_type='BRCA', eligible_for_somatic_status=0) for source in ('CGC', 'OncoKB', 'IntOGen')]
        statuses = build_status(records, [{'gene_ens': GENE, 'gene_symbol': 'A'}], {GENE}, [], set())
        self.assertEqual(statuses[0]['evidence_status'], 'unresolved_candidate')
        self.assertEqual(statuses[0]['benchmark_label'], 0)
        self.assertEqual(statuses[0]['negative_label_status'], 'unlabelled_not_confirmed_negative')

    def test_reviewed_somatic_support_keeps_original_zero(self):
        row = evidence_row(mapper().resolve('A'), source_resource='PrimaryLiterature',
            source_record_id='study:A', assertion='reviewed_somatic_BRCA_driver_evidence',
            setting='somatic', eligible_for_somatic_status=1, study_id='DOI:study')
        statuses = build_status([row], [{'gene_ens': GENE, 'gene_symbol': 'A'}], {GENE}, [], set())
        self.assertEqual(statuses[0]['evidence_status'], 'supported_somatic_BRCA_driver')
        self.assertEqual(statuses[0]['benchmark_label'], 0)
        validate_tables([row], statuses, set())

    def test_scope_unknown_cgc_subset_keeps_supplied_positive(self):
        row = evidence_row(mapper().resolve('A'), source_resource='CGC', source_record_id='A',
            assertion='historical_benchmark_positive', setting='not_specified', eligible_for_somatic_status=0)
        statuses = build_status([row], [{'gene_ens': GENE, 'gene_symbol': 'A'}], {GENE}, [], {GENE})
        self.assertEqual(statuses[0]['benchmark_label'], 1)
        self.assertEqual(statuses[0]['cgc_v95_subset_member'], 1)
        self.assertEqual(statuses[0]['evidence_status'], 'unresolved_candidate')
        validate_tables([row], statuses, {GENE})

    def test_absence_filtered_calls_and_limited_negative_evidence_are_not_negatives(self):
        row = evidence_row(mapper().resolve('A'), source_resource='PrimaryLiterature', source_record_id='A',
            assertion='limited_negative_germline_association_evidence', setting='germline',
            eligible_for_somatic_status=0)
        statuses = build_status([row], [{'gene_ens': gene, 'gene_symbol': symbol}
            for gene, symbol in ((GENE, 'A'), (OTHER, 'B'))], {GENE, OTHER}, [], set())
        self.assertEqual(statuses[1]['evidence_status'], 'insufficient_evidence')
        self.assertTrue(all(row['negative_label_status'] == 'unlabelled_not_confirmed_negative' for row in statuses))
        self.assertIn('limited_germline_negative_evidence', statuses[0]['conflicting_or_filtered_evidence'])

    def test_broken_or_duplicate_links_fail(self):
        row = evidence_row(mapper().resolve('A'), source_resource='OncoKB', source_record_id='A')
        statuses = build_status([row], [], set(), [], set())
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_tables([row, row], statuses, set())
        statuses[0]['source_record_count'] = 2
        with self.assertRaisesRegex(ValueError, 'links'):
            validate_tables([row], statuses, set())

    def test_bad_row_widths_fail(self):
        with self.assertRaisesRegex(ValueError, 'widths'):
            read_table(io.StringIO('A\tB\n1\n'), '\t', ('A', 'B'))

    def test_source_version_disagreement_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.zip'
            with ZipFile(path, 'w') as archive:
                archive.writestr('2024-06-18_Drivers/README.txt', '# IntOGen RELEASE 20230531\n')
            version = archive_version(path)
            self.assertEqual(version['archive_directory'], '2024-06-18_Drivers')
            self.assertEqual(version['readme_release'], '20230531')

    def test_feature_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'BRCA_input_data.csv').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'separate'):
                run_driver_evidence({}, cache=None, output_dir=path, canonical_path='', reviewed_path='')
            self.assertEqual((path / 'BRCA_input_data.csv').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
