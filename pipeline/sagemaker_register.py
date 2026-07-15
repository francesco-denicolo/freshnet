"""
sagemaker_register.py — registra l'output di un training job nel SageMaker Model Registry
=========================================================================================
Crea (se manca) il Model Package Group e vi registra una nuova **versione** che punta al
model.tar.gz prodotto dal job, con le metriche finali (raccolte da CloudWatch via le
MetricDefinitions del job) come metadata.

Uso:
    aws-vault exec evocity -- python pipeline/sagemaker_register.py --job <job-name>
    aws-vault exec evocity -- python pipeline/sagemaker_register.py --list
    aws-vault exec evocity -- python pipeline/sagemaker_register.py --approve <model-package-arn>

Nota: il Model Registry richiede sempre una InferenceSpecification (immagine + artefatti)
anche quando il modello non viene deployato: usiamo la stessa immagine del training, così
la versione registrata è auto-contenuta e riproducibile.
"""
import argparse, json, os, sys
import boto3

REGION = os.getenv('AWS_REGION', 'eu-central-1')
GROUP = os.getenv('TFT_MODEL_GROUP', 'tft-forecasting')


def ensure_group(sm, group):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group)
        print(f'Model Package Group esistente: {group}')
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription='TFT (Temporal Fusion Transformer) su FreshRetailNet-50K '
                                         '— benchmark imputer x forecaster',
        )
        print(f'Model Package Group creato: {group}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', help='nome del training job da registrare')
    p.add_argument('--group', default=GROUP)
    p.add_argument('--approve', help='ARN di un model package da approvare')
    p.add_argument('--list', action='store_true', help='elenca le versioni registrate')
    p.add_argument('--approved', action='store_true',
                   help='registra già come Approved (default: PendingManualApproval)')
    a = p.parse_args()

    sm = boto3.client('sagemaker', region_name=REGION)

    if a.approve:
        sm.update_model_package(ModelPackageArn=a.approve, ModelApprovalStatus='Approved')
        print('Approvato:', a.approve)
        return

    if a.list:
        r = sm.list_model_packages(ModelPackageGroupName=a.group, SortBy='CreationTime',
                                   SortOrder='Descending', MaxResults=50)
        for m in r.get('ModelPackageSummaryList', []):
            print(f"v{m['ModelPackageVersion']:>3}  {m['ModelApprovalStatus']:<22} "
                  f"{m['CreationTime']:%Y-%m-%d %H:%M}  {m['ModelPackageArn']}")
        if not r.get('ModelPackageSummaryList'):
            print(f'(nessuna versione in {a.group})')
        return

    if not a.job:
        sys.exit('Serve --job <training-job-name> (oppure --list / --approve)')

    d = sm.describe_training_job(TrainingJobName=a.job)
    status = d['TrainingJobStatus']
    if status != 'Completed':
        sys.exit(f'Il job {a.job} è in stato {status}: registro solo i job Completed.')

    artifacts = d['ModelArtifacts']['S3ModelArtifacts']
    image = d['AlgorithmSpecification']['TrainingImage']
    hp = d.get('HyperParameters', {})

    # metriche finali raccolte da SageMaker via le MetricDefinitions (-> CloudWatch)
    metrics = {m['MetricName']: round(float(m['Value']), 4)
               for m in d.get('FinalMetricDataList', [])}
    print('Artefatti :', artifacts)
    print('Immagine  :', image)
    print('Metriche  :', metrics or '(nessuna: MetricDefinitions non hanno matchato)')

    ensure_group(sm, a.group)

    meta = {'training_job': a.job,
            'instance': d['ResourceConfig']['InstanceType'],
            'spot': str(d.get('EnableManagedSpotTraining', False)),
            'series_subsample': str(hp.get('SERIES_SUBSAMPLE', '0')),
            'hidden_cap': str(hp.get('TFT_HIDDEN_CAP', '')),
            'precision': str(hp.get('TFT_PRECISION', '')),
            **{f'metric_{k.replace(":", "_")}': str(v) for k, v in metrics.items()}}

    resp = sm.create_model_package(
        ModelPackageGroupName=a.group,
        ModelPackageDescription=f'TFT da {a.job} — subsample={meta["series_subsample"]}, '
                                f'hidden_cap={meta["hidden_cap"]}',
        InferenceSpecification={
            'Containers': [{'Image': image, 'ModelDataUrl': artifacts}],
            'SupportedContentTypes': ['application/x-parquet'],
            'SupportedResponseMIMETypes': ['application/json'],
        },
        ModelApprovalStatus='Approved' if a.approved else 'PendingManualApproval',
        CustomerMetadataProperties=meta,
    )
    arn = resp['ModelPackageArn']
    print(f'\nRegistrato nel Model Registry: {arn}')
    print(f'  group  : {a.group}')
    print(f'  status : {"Approved" if a.approved else "PendingManualApproval"}')
    print(f'\nApprova con: python pipeline/sagemaker_register.py --approve {arn}')


if __name__ == '__main__':
    main()
