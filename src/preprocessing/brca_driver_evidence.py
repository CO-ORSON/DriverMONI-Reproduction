"""Versioned oncology evidence and BRCA review status, separate from model labels."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

from .acquisition import open_text_source
from .brca_sources import normalize_ensembl, read_labels
from .common import sha256_file, utc_now, write_csv, write_json


EVIDENCE_FIELDS = (
    "evidence_id", "gene_key", "gene_ens", "gene_symbol", "source_resource", "source_version",
    "source_record_id", "source_artifact", "source_url", "source_sha256", "source_date",
    "source_symbol", "source_gene_id", "mapping_status", "cancer_type", "cancer_context",
    "setting", "alteration_mechanism", "assertion", "evidence_strength", "source_filter",
    "source_tier", "oncokb_oncogenicity", "oncokb_therapeutic_level", "cohort_id",
    "cohort_source", "cohort_type", "cohort_treatment", "cohort_overlap", "significance",
    "methods", "study_id", "citations", "independence_group", "eligible_for_somatic_status",
    "evidence_note", "source_payload_json",
)
STATUS_FIELDS = (
    "gene_key", "gene_ens", "gene_symbol", "entity_type", "mapping_status", "benchmark_label",
    "in_original_features", "in_benchmark_evaluation", "in_canonical_features",
    "evidence_status", "brca_evidence_scope", "review_basis", "germline_evidence_status", "source_resources",
    "source_record_count", "evidence_ids", "reviewed_somatic_study_count",
    "intogen_brca_pass_cohorts", "intogen_primary_untreated_pass_cohorts",
    "intogen_same_project_or_possible_overlap_cohorts", "oncokb_breast_summary_mentions",
    "cgc_v95_subset_member", "independence_status", "mechanism_coverage",
    "conflicting_or_filtered_evidence", "negative_label_status", "review_required",
)
PROFILE = "BRCA_driver_evidence_v1"


def read_table(handle, delimiter, required):
    reader = csv.DictReader(handle, delimiter=delimiter)
    if not set(required) <= set(reader.fieldnames or []):
        raise ValueError(f"Evidence source requires columns {sorted(required)}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Evidence table contains incompatible row widths")
    return rows


def zip_table(path, suffix, required):
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('/' + suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one {suffix} in source archive")
        with archive.open(names[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig") as handle:
            return read_table(handle, '\t', required), names[0]


def archive_version(path):
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('/README.txt')]
        if len(names) != 1:
            raise ValueError("Archive has no unique version README")
        text = archive.read(names[0]).decode('utf-8-sig')
        match = re.search(r'IntOGen RELEASE (\d{8})', text)
        return {"archive_directory": names[0].split('/')[0],
                "readme_release": match.group(1) if match else "not_declared"}


class GeneMapper:
    """Require unique HGNC/NCBI/Ensembl mappings; never choose the first match."""

    def __init__(self, hgnc, transcripts, annotation_symbols):
        self.by_symbol, self.by_ncbi = defaultdict(list), defaultdict(list)
        self.by_gene = defaultdict(set)
        self.transcripts, self.annotation_symbols = transcripts, annotation_symbols
        for row in hgnc:
            if row['status'] != 'Approved' or not re.fullmatch(r'ENSG\d{11}', row['ensembl_gene_id']):
                continue
            self.by_symbol[row['symbol']].append(row)
            if row['entrez_id']:
                self.by_ncbi[row['entrez_id']].append(row)
            self.by_gene[row['ensembl_gene_id']].add(row['symbol'])

    def resolve(self, symbol, ncbi='', transcript=''):
        by_symbol = self.by_symbol.get(symbol, [])
        by_ncbi = self.by_ncbi.get(str(ncbi), []) if ncbi else []
        symbol_genes = {row['ensembl_gene_id'] for row in by_symbol}
        ncbi_genes = {row['ensembl_gene_id'] for row in by_ncbi}
        if symbol_genes and ncbi_genes and symbol_genes != ncbi_genes:
            return self.unresolved(symbol, 'symbol_ncbi_conflict')
        rows = by_ncbi or by_symbol
        choices = ncbi_genes or symbol_genes
        method = 'unique_hgnc_ncbi' if by_ncbi else 'unique_hgnc_symbol'
        if not choices:
            choices = set(self.annotation_symbols.get(symbol, set()))
            method = 'unique_ensembl102_symbol'
        transcript_genes = set(self.transcripts.get(transcript.split('.')[0], set())) if transcript else set()
        if transcript_genes and choices and transcript_genes != choices:
            return self.unresolved(symbol, 'transcript_gene_conflict')
        if not choices and transcript_genes:
            choices = transcript_genes
            method = 'unique_ensembl102_transcript'
        if len(choices) != 1:
            return self.unresolved(symbol, 'ambiguous_mapping' if choices else 'unmapped')
        gene = next(iter(choices))
        approved = {row['symbol'] for row in rows if row['ensembl_gene_id'] == gene}
        if not approved:
            approved = self.by_gene.get(gene, set())
        if len(approved) > 1:
            return self.unresolved(symbol, 'ambiguous_approved_symbol')
        if transcript and not transcript_genes:
            method += ';transcript_not_in_release102'
        return {'gene_key': gene, 'gene_ens': gene, 'gene_symbol': next(iter(approved), symbol),
                'mapping_status': method}

    @staticmethod
    def unresolved(symbol, status):
        return {'gene_key': 'UNRESOLVED_SYMBOL:' + symbol, 'gene_ens': '',
                'gene_symbol': symbol, 'mapping_status': status}


def annotation_mappings(handle):
    transcripts, symbols = defaultdict(set), defaultdict(set)
    for line in handle:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) != 9:
            raise ValueError("Malformed Ensembl annotation")
        if fields[2] not in ('gene', 'transcript'):
            continue
        attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
        gene = normalize_ensembl(attrs['gene_id'])
        if fields[2] == 'gene' and attrs.get('gene_name'):
            symbols[attrs['gene_name']].add(gene)
        if fields[2] == 'transcript':
            transcripts[attrs['transcript_id'].split('.')[0]].add(gene)
    return dict(transcripts), dict(symbols)


def pmids(text):
    # Only explicitly labelled PMID blocks, never arbitrary numeric tokens.
    return '|'.join('PMID:' + value for value in sorted({
        item for block in re.findall(r'PMID:\s*([\d,\s]+)', text)
        for item in re.findall(r'\d{5,9}', block)}))


def mechanism_mentions(text):
    patterns = {
        'SNV_indel': r'\bmutat(?:ion|ions|ed)\b',
        'copy_number': r'\bamplif\w*|\bdelet\w*|\bcopy.number\b',
        'fusion_rearrangement': r'\bfusion\w*|\brearrang\w*',
        'expression': r'\b(?:overexpress\w*|downregulat\w*|expression)\b',
    }
    return '|'.join(name for name, pattern in patterns.items() if re.search(pattern, text, re.I)) or 'not_specified'


def evidence_row(mapping, **values):
    row = dict.fromkeys(EVIDENCE_FIELDS, '')
    row.update(mapping)
    row.update(values)
    if isinstance(row['source_payload_json'], dict):
        row['source_payload_json'] = json.dumps(row['source_payload_json'], sort_keys=True, ensure_ascii=False)
    identity = [row[key] for key in ('gene_key', 'source_resource', 'source_version', 'source_record_id',
                                    'setting', 'cohort_id', 'assertion', 'study_id')]
    row['evidence_id'] = 'EVIDENCE_' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    return row


def intogen_evidence(compendium, raw, cohorts, mapper, metadata):
    required = ('SYMBOL', 'COHORT', 'TRANSCRIPT')
    accepted = {tuple(row[k] for k in required): row for row in compendium if row['CANCER_TYPE'] == 'BRCA'}
    if len(accepted) != sum(row['CANCER_TYPE'] == 'BRCA' for row in compendium):
        raise ValueError("Duplicate IntOGen gene/transcript/cohort assertion")
    results, seen = [], set()
    for row in raw:
        if row['CANCER_TYPE'] != 'BRCA':
            continue
        key = tuple(row[k] for k in required)
        if key in seen:
            raise ValueError("Duplicate unfiltered IntOGen assertion")
        seen.add(key)
        cohort = cohorts.get(row['COHORT'])
        if not cohort or cohort['CANCER'] != 'BRCA':
            raise ValueError("IntOGen BRCA assertion lacks compatible cohort metadata")
        passing = row['DRIVER'] == 'True' and row['FILTER'] == 'PASS'
        if passing != (key in accepted):
            raise ValueError("IntOGen compendium disagrees with DRIVER/FILTER")
        overlap = ('same_project_possible_sample_overlap' if 'TCGA' in cohort['COHORT'] or
                   'Cancer Genome Atlas' in cohort['SOURCE'] else
                   'possible_TCGA_ICGC_overlap' if 'PCAWG' in cohort['SOURCE'] else
                   'sample_overlap_not_verified')
        results.append(evidence_row(mapper.resolve(row['SYMBOL'], transcript=row['TRANSCRIPT']),
            **metadata, source_record_id='|'.join(key), source_symbol=row['SYMBOL'],
            source_gene_id=row['TRANSCRIPT'], cancer_type='BRCA', cancer_context=cohort['CANCER_NAME'],
            setting='somatic', alteration_mechanism='SNV_indel',
            assertion='BRCA_somatic_driver_call' if passing else 'BRCA_unaccepted_driver_call',
            evidence_strength='filtered_cohort_statistical_evidence' if passing else 'filtered_out_or_not_significant',
            source_filter=row['FILTER'], source_tier=row['TIER'], cohort_id=row['COHORT'],
            cohort_source=cohort['SOURCE'], cohort_type=cohort['TYPE'], cohort_treatment=cohort['TREATED'],
            cohort_overlap=overlap, significance=row['QVALUE_COMBINATION'], methods=row['ALL_METHODS'],
            study_id=cohort['REFERENCE'], citations=cohort['REFERENCE'],
            independence_group='IntOGen:' + cohort['SOURCE'], eligible_for_somatic_status=0,
            evidence_note='Cohort call retained for review; statistical acceptance alone does not assign a somatic-driver status. DRIVER=True with FILTER other than PASS is not accepted. Cohorts can share samples and curated literature.',
            source_payload_json={'unfiltered': row, 'compendium': accepted.get(key), 'cohort': cohort,
                                 'cohort_metadata_manifest_role': 'intogen_cohorts'}))
    if not set(accepted) <= seen:
        raise ValueError("IntOGen compendium contains assertions absent from the raw table")
    return results


def build_status(evidence, originals, evaluated, canonical, labels):
    grouped, identities = defaultdict(list), {}
    for row in originals + canonical:
        gene = normalize_ensembl(row['gene_ens'])
        identities[gene] = {'gene_key': gene, 'gene_ens': gene, 'gene_symbol': row['gene_symbol'],
                            'mapping_status': 'existing_feature_identifier'}
    for row in evidence:
        grouped[row['gene_key']].append(row)
        if row['gene_key'] not in identities or row['mapping_status'].startswith('unique_'):
            identities[row['gene_key']] = {key: row[key] for key in ('gene_key', 'gene_ens', 'gene_symbol', 'mapping_status')}
    original_genes = {normalize_ensembl(row['gene_ens']) for row in originals}
    canonical_genes = {normalize_ensembl(row['gene_ens']) for row in canonical}
    results = []
    for key in sorted(identities):
        rows, identity = grouped[key], identities[key]
        gene = identity['gene_ens']
        supported = [row for row in rows if str(row['eligible_for_somatic_status']) == '1']
        accepted = [row for row in rows if row['assertion'] == 'BRCA_somatic_driver_call']
        germline = [row for row in rows if row['setting'] == 'germline' and
                    (row['assertion'] == 'reviewed_germline_breast_risk_evidence' or row['cancer_type'] == 'BRCA')]
        filtered = [row for row in rows if row['assertion'] == 'BRCA_unaccepted_driver_call']
        negative = [row for row in rows if row['assertion'] == 'limited_negative_germline_association_evidence']
        status = ('unresolved_candidate' if not gene else 'supported_somatic_BRCA_driver' if supported else
                  'unresolved_candidate' if rows else 'insufficient_evidence')
        basis = ('Reviewed published somatic BRCA evidence; not a new training label' if supported else
                 'Mapping unresolved; no gene-level adjudication' if not gene else
                 'Source evidence requires BRCA/mechanism/setting review' if rows else
                 'No evidence in the acquired catalogues or current literature spot-check; not a negative')
        mentions = [row for row in rows if row['source_resource'] == 'OncoKB' and row['cancer_type'] == 'BRCA']
        scopes = set()
        if supported:
            scopes.add('reviewed_somatic_BRCA')
        if gene in labels:
            scopes.add('paper_attributed_CGC_v95_BRCA_subset')
        if any(row['source_resource'] == 'IntOGen' for row in rows):
            scopes.add('BRCA_cohort_statistical_results')
        if germline:
            scopes.add('germline_breast_context')
        if mentions:
            scopes.add('unreviewed_OncoKB_breast_summary_mention')
        if not scopes:
            scopes.add('pan_cancer_only' if rows else 'no_acquired_evidence')
        results.append(dict(identity, entity_type='resolved_gene' if gene else 'unresolved_source_entity',
            benchmark_label=1 if gene in labels else 0 if gene in original_genes else '',
            in_original_features=int(gene in original_genes), in_benchmark_evaluation=int(gene in evaluated),
            in_canonical_features=int(gene in canonical_genes), evidence_status=status,
            brca_evidence_scope='|'.join(sorted(scopes)), review_basis=basis,
            germline_evidence_status='reviewed_breast_risk_evidence' if any(row['assertion'] == 'reviewed_germline_breast_risk_evidence' for row in germline) else
                'source_breast_mention_requires_review' if germline else 'not_established',
            source_resources='|'.join(sorted({row['source_resource'] for row in rows})), source_record_count=len(rows),
            evidence_ids='|'.join(sorted(row['evidence_id'] for row in rows)),
            reviewed_somatic_study_count=len({row['study_id'] for row in supported}),
            intogen_brca_pass_cohorts=len({row['cohort_id'] for row in accepted}),
            intogen_primary_untreated_pass_cohorts=len({row['cohort_id'] for row in accepted if row['cohort_type'] == 'Primary' and row['cohort_treatment'] == 'Untreated'}),
            intogen_same_project_or_possible_overlap_cohorts=len({row['cohort_id'] for row in accepted if row['cohort_overlap'] in ('same_project_possible_sample_overlap', 'possible_TCGA_ICGC_overlap')}),
            oncokb_breast_summary_mentions=len(mentions), cgc_v95_subset_member=int(gene in labels),
            independence_status='shared_literature_and_sample_overlap_not_fully_adjudicated' if rows else 'not_assessed',
            mechanism_coverage='SNV/indel consequence and expression summaries available; no dedicated copy-number, fusion or germline features' if gene in canonical_genes or gene in original_genes else 'no_existing_feature_row',
            conflicting_or_filtered_evidence='|'.join(sorted({row['source_filter'] for row in filtered} | {'limited_germline_negative_evidence' for _ in negative})),
            negative_label_status='supplied_positive' if gene in labels else 'unlabelled_not_confirmed_negative',
            review_required=1))
    return results


def validate_tables(evidence, statuses, labels):
    ids = [row['evidence_id'] for row in evidence]
    keys = [row['gene_key'] for row in statuses]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("Evidence IDs and gene keys must be unique")
    by_id, by_key = {row['evidence_id']: row for row in evidence}, {row['gene_key']: row for row in statuses}
    for row in evidence:
        if row['gene_key'] not in by_key:
            raise ValueError("Evidence references an absent gene-status row")
    for row in statuses:
        links = row['evidence_ids'].split('|') if row['evidence_ids'] else []
        if len(links) != row['source_record_count'] or any(by_id[link]['gene_key'] != row['gene_key'] for link in links):
            raise ValueError("Gene-status evidence links disagree with source records")
        if row['evidence_status'] == 'supported_somatic_BRCA_driver' and not any(
                str(by_id[link]['eligible_for_somatic_status']) == '1' and by_id[link]['setting'] == 'somatic' for link in links):
            raise ValueError("Somatic status requires explicitly reviewed somatic evidence")
        if row['negative_label_status'] not in ('supplied_positive', 'unlabelled_not_confirmed_negative'):
            raise ValueError("This catalogue must not manufacture negative labels")
    if {row['gene_ens'] for row in statuses if row['benchmark_label'] == 1} != set(labels):
        raise ValueError("Benchmark positives changed during evidence preparation")


def run_driver_evidence(config, *, cache, output_dir, canonical_path, reviewed_path, lock=None):
    output = Path(output_dir)
    if (output / 'BRCA_input_data.csv').exists():
        raise ValueError("Evidence output must be separate from a feature dataset")
    if (output / 'manifest.json').exists() and json.loads((output / 'manifest.json').read_text()).get('profile') != PROFILE:
        raise ValueError("Evidence output belongs to a different profile")
    paths = {}
    for role, spec in config['sources'].items():
        print(f'Verifying/acquiring {role}...', flush=True)
        paths[role] = cache.get(role, spec)
        if spec.get('sha256') and sha256_file(paths[role]) != spec['sha256']:
            raise ValueError(f'Source differs from configured snapshot: {role}')
    input_records = [{'role': role, 'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}
                     for role, path in [('canonical_features', canonical_path), ('reviewed_assertions', reviewed_path)]]
    if lock is not None:
        prior = {row['role']: row['sha256'] for row in lock['inputs']}
        if any(prior.get(row['role']) != row['sha256'] for row in input_records):
            raise ValueError('Local evidence inputs differ from the locked manifest')
    def table(role, required, delimiter='\t'):
        with open_text_source(paths[role], config['sources'][role]) as handle:
            return read_table(handle, delimiter, required)
    hgnc = table('hgnc', ('symbol', 'status', 'ensembl_gene_id', 'entrez_id'))
    with open_text_source(paths['ensembl'], config['sources']['ensembl']) as handle:
        transcripts, symbols = annotation_mappings(handle)
    mapper = GeneMapper(hgnc, transcripts, symbols)
    originals = table('benchmark_features', ('gene_ens', 'gene_symbol'), ',')
    original_by_gene = {row['gene_ens']: row for row in originals}
    labels = set(read_labels(paths['benchmark_labels']))
    comparisons = table('comparison_genes', ('gene',), ',')
    annotations = table('gene_annotations', ('Symbol',), ',')
    allowed_symbols = {row['Symbol'] for row in annotations}
    if any(row['gene'] not in original_by_gene for row in comparisons):
        raise ValueError('Comparison gene absent from original benchmark')
    comparison_symbols = {original_by_gene[row['gene']]['gene_symbol'] for row in comparisons}
    evaluated = {row['gene_ens'] for row in originals if row['gene_symbol'] in comparison_symbols & allowed_symbols}
    with Path(canonical_path).open(encoding='utf-8-sig', newline='') as handle:
        canonical = read_table(handle, ',', ('gene_ens', 'gene_symbol'))
    info = json.loads(paths['oncokb_metadata'].read_text())
    onco_version, onco_date = info['dataVersion']['version'], info['dataVersion']['date']
    versions = {role: dict(archive_version(paths[role]), advertised_release=config['sources'][role]['advertised_release'])
                for role in ('intogen_drivers', 'intogen_cohorts')}
    def metadata(role, resource, version, date=''):
        return {'source_resource': resource, 'source_version': version, 'source_url': config['sources'][role]['url'],
                'source_artifact': role, 'source_sha256': sha256_file(paths[role]), 'source_date': date}
    evidence = []
    for gene in sorted(labels):
        if gene not in original_by_gene:
            raise ValueError('Supplied driver has no original feature identifier')
        identity = {'gene_key': gene, 'gene_ens': gene, 'gene_symbol': original_by_gene[gene]['gene_symbol'],
                    'mapping_status': 'supplied_Ensembl_ID'}
        evidence.append(evidence_row(identity, **metadata('benchmark_labels', 'CGC', 'v95_attributed_DriverMONI_subset'),
            source_record_id=gene, source_symbol=identity['gene_symbol'], source_gene_id=gene,
            cancer_type='BRCA', cancer_context='DriverMONI supplied BRCA driver list', setting='not_specified',
            assertion='historical_benchmark_positive', evidence_strength='paper_attributed_subset_unverified_extraction',
            citations='DOI:10.1109/TCBBIO.2025.3643887', independence_group='DriverMONI_CGC_attribution',
            eligible_for_somatic_status=0,
            evidence_note='Used at user request because the full CGC v95 export is unavailable. These are the 42 DriverMONI labels, not a verified CGC export; tier and exact somatic/germline BRCA extraction rules are unknown.'))
    gene_list = table('oncokb_gene_list', ('Hugo Symbol', 'Entrez Gene ID', 'Gene Type', 'OncoKB Annotated'))
    for row in gene_list:
        evidence.append(evidence_row(mapper.resolve(row['Hugo Symbol'], row['Entrez Gene ID']),
            **metadata('oncokb_gene_list', 'OncoKB', onco_version, onco_date),
            source_record_id=row['Entrez Gene ID'], source_symbol=row['Hugo Symbol'], source_gene_id=row['Entrez Gene ID'],
            cancer_type='pan_cancer', cancer_context='Broad cancer-gene/panel list; BRCA association not supplied',
            setting='not_specified', assertion='broad_cancer_gene_list_membership', evidence_strength='candidate_context_only',
            independence_group='OncoKB_CGC_shared_curation', eligible_for_somatic_status=0,
            evidence_note='Panel/list membership and the embedded CGC v99 flag do not establish a somatic BRCA driver or independent CGC v95 evidence.',
            source_payload_json=row))
    curated = table('oncokb_curated_genes', ('Hugo Symbol', 'Entrez Gene ID', 'Gene Type', 'Setting', 'Summary', 'Background', 'Highest Level of Evidence(sensitivity)'))
    for row in curated:
        setting = row['Setting'].lower()
        if setting not in ('somatic', 'germline'):
            raise ValueError('Unexpected OncoKB curation setting')
        breast = bool(re.search(r'\bbreast\b', row['Summary'], re.I))
        evidence.append(evidence_row(mapper.resolve(row['Hugo Symbol'], row['Entrez Gene ID']),
            **metadata('oncokb_curated_genes', 'OncoKB', onco_version, onco_date),
            source_record_id=row['Entrez Gene ID'] + ':' + setting, source_symbol=row['Hugo Symbol'], source_gene_id=row['Entrez Gene ID'],
            cancer_type='BRCA' if breast else 'pan_cancer', cancer_context='Unreviewed breast mention in gene summary' if breast else 'Gene-level curation; cancer context not resolved',
            setting=setting, alteration_mechanism=mechanism_mentions(row['Summary']),
            assertion='curated_gene_context', evidence_strength='curated_gene_summary_requires_context_review',
            oncokb_therapeutic_level=row['Highest Level of Evidence(sensitivity)'], citations=pmids(row['Summary'] + ' ' + row['Background']),
            independence_group='OncoKB_CGC_shared_curation', eligible_for_somatic_status=0,
            evidence_note='Summary keyword matching only flags review candidates; it is not a BRCA assertion or oncogenicity annotation. Somatic-setting records can discuss germline susceptibility. Therapeutic levels are not used to decide driver status. Citations belong to the full gene background and are not individually adjudicated as BRCA evidence.',
            source_payload_json=row))
    compendium, compendium_member = zip_table(paths['intogen_drivers'], 'Compendium_Cancer_Genes.tsv', ('SYMBOL', 'TRANSCRIPT', 'COHORT', 'CANCER_TYPE'))
    raw, raw_member = zip_table(paths['intogen_drivers'], 'Unfiltered_drivers.tsv', ('SYMBOL', 'TRANSCRIPT', 'COHORT', 'CANCER_TYPE', 'DRIVER', 'FILTER'))
    cohort_rows, cohort_member = zip_table(paths['intogen_cohorts'], 'cohorts.tsv', ('COHORT', 'CANCER', 'CANCER_NAME', 'SOURCE', 'TYPE', 'TREATED', 'REFERENCE'))
    cohorts = {row['COHORT']: row for row in cohort_rows}
    if len(cohorts) != len(cohort_rows):
        raise ValueError('Duplicate IntOGen cohort identifiers')
    intogen_meta = metadata('intogen_drivers', 'IntOGen',
                            'advertised_' + versions['intogen_drivers']['advertised_release'] + '_pinned_archive')
    intogen_meta['source_artifact'] = ('intogen_drivers:' + raw_member + ' + ' + compendium_member +
                                     ' + intogen_cohorts:' + cohort_member)
    evidence.extend(intogen_evidence(compendium, raw, cohorts, mapper, intogen_meta))
    reviewed = json.loads(Path(reviewed_path).read_text())
    for assertion in reviewed['assertions']:
        if assertion['setting'] not in ('somatic', 'germline'):
            raise ValueError('Reviewed assertion needs an explicit biological setting')
        for symbol in assertion['gene_symbols']:
            identity = mapper.resolve(symbol)
            eligible = (assertion['assertion'] == 'reviewed_somatic_BRCA_driver_evidence' and
                        assertion['setting'] == 'somatic' and bool(identity['gene_ens']))
            evidence.append(evidence_row(identity, source_resource='PrimaryLiterature', source_version=reviewed['review_version'],
                source_record_id=assertion['study_id'] + ':' + symbol, source_artifact='reviewed_assertions',
                source_url=assertion['source_url'], source_sha256=sha256_file(reviewed_path),
                source_date=assertion['publication_year'], source_symbol=symbol, cancer_type='BRCA',
                cancer_context='Reviewed breast-cancer study assertion', setting=assertion['setting'],
                alteration_mechanism=assertion['alteration_mechanism'], assertion=assertion['assertion'],
                evidence_strength='published_gene_level_evidence_spot_check', cohort_id=assertion['cohort_id'],
                cohort_overlap=assertion['cohort_overlap'], study_id=assertion['study_id'], citations=assertion['study_id'],
                independence_group=assertion['cohort_id'], eligible_for_somatic_status=int(eligible),
                evidence_note=assertion['review_note'] + ' Hash covers the local reviewed-assertion table, not the journal webpage.',
                source_payload_json=assertion))
    evidence.sort(key=lambda row: (row['gene_key'], row['source_resource'], row['source_record_id']))
    statuses = build_status(evidence, originals, evaluated, canonical, labels)
    validate_tables(evidence, statuses, labels)
    summary = {
        'profile': PROFILE, 'evidence_rows': len(evidence), 'gene_status_rows': len(statuses),
        'resolved_gene_rows': sum(bool(row['gene_ens']) for row in statuses),
        'unresolved_source_entities': sum(not row['gene_ens'] for row in statuses),
        'source_record_counts': dict(Counter(row['source_resource'] for row in evidence)),
        'evidence_status_counts': dict(Counter(row['evidence_status'] for row in statuses)),
        'benchmark_positive_genes': len(labels), 'benchmark_evaluated_genes': len(evaluated),
        'benchmark_unlabelled_genes': len(evaluated - labels),
        'supported_genes_with_benchmark_zero': [row['gene_symbol'] for row in statuses if row['evidence_status'] == 'supported_somatic_BRCA_driver' and row['benchmark_label'] == 0],
        'unresolved_mapping_records': sum(not row['gene_ens'] for row in evidence),
        'mapping_status_counts': dict(Counter(row['mapping_status'] for row in evidence)),
        'intogen_filtered_BRCA_records': sum(row['assertion'] == 'BRCA_somatic_driver_call' for row in evidence),
        'intogen_filtered_BRCA_genes': len({row['gene_key'] for row in evidence if row['assertion'] == 'BRCA_somatic_driver_call'}),
        'versions': {'OncoKB': info['dataVersion'], 'IntOGen': versions, 'HGNC': config['sources']['hgnc']['release'], 'Ensembl': 102},
        'coverage': {
            'CGC': '42 DriverMONI positives used as paper-attributed v95 subset at user request; full export unavailable, extraction and tiers unverified',
            'OncoKB': 'Full public gene-list and somatic/germline curated-gene exports; tumour/alteration-level export requires authorization and was not acquired',
            'IntOGen': 'Complete downloaded compendium, unfiltered assertions and cohort metadata; BRCA records selected by exact cancer code',
            'literature': reviewed['scope'],
        },
        'limits': [
            'IntOGen advertised filename, archive directory and README release disagree; all labels and hashes are preserved rather than silently choosing a version.',
            'Gene-status decisions are evidence reviews, not a new training-label export; all original positives remain unchanged.',
            'No confirmed gene-level negatives are asserted. Rejected IntOGen calls and germline non-association findings are not somatic negative labels.',
            'Shared publications and patient overlap are not fully resolved; database/cohort counts are not counts of independent validations.',
            'OncoKB summary mentions and therapeutic levels do not establish alteration oncogenicity or somatic BRCA specificity.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'brca_driver_evidence.csv', EVIDENCE_FIELDS, evidence)
    write_csv(output / 'brca_gene_status.csv', STATUS_FIELDS, statuses)
    write_json(output / 'summary.json', summary)
    manifest = {'schema_version': 1, 'profile': PROFILE, 'created_at': utc_now(), 'configuration': config,
        'sources': list(cache.sources.values()), 'inputs': input_records,
        'policy': {'benchmark': 'Preserve DriverMONI 42 positives as the attributed CGC v95 subset; do not claim exact historical extraction',
            'mapping': 'Unique approved HGNC symbol/NCBI gene mapping; check transcripts against Ensembl102; conflicts and ambiguity remain unresolved',
            'somatic_status': 'Only explicitly reviewed published somatic BRCA assertions qualify; other database records remain candidates',
            'germline': 'Separate field; never promotes somatic status',
            'negatives': 'No absence/rejection/non-association rule assigns a gene-level negative',
            'independence': 'Preserve studies/cohorts and overlap flags; no database voting',
            'IntOGen_acceptance': 'Exact BRCA cancer code; DRIVER=True AND FILTER=PASS with matching compendium record'},
        'implementation': {'code_sha256': sha256_file(__file__)},
        'outputs': [{'file': name, 'sha256': sha256_file(output / name)} for name in
                    ('brca_driver_evidence.csv', 'brca_gene_status.csv', 'summary.json')]}
    write_json(output / 'manifest.json', manifest)
    return summary
