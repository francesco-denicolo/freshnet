"""
sagemaker_launch.py — lancia il TFT come SageMaker Training Job (GPU + checkpoint/resume)
=========================================================================================
Uso (con credenziali via aws-vault):
    aws-vault exec evocity -- python pipeline/sagemaker_launch.py \
        --image <ACCOUNT>.dkr.ecr.eu-central-1.amazonaws.com/tft-gpu:latest \
        --role  arn:aws:iam::<ACCOUNT>:role/<SageMakerExecutionRole>

Opzioni utili:
    --spot                 usa managed spot (~70% di risparmio; il resume è automatico
                           grazie a CheckpointConfig -> /opt/ml/checkpoints)
    --subsample 15000      riduce le serie (default 0 = tutte le 50K)
    --instance ml.g4dn.2xlarge
    --status <job-name>    mostra solo lo stato di un job esistente

Perché Training Job (e non Processing): SageMaker sincronizza /opt/ml/checkpoints su S3
DURANTE il run, quindi un'interruzione Spot non perde nulla e il job riprende da dove era
(Optuna riprende dal DB, le celle già fatte vengono saltate).
"""
import argparse, os, sys, time
import boto3

BUCKET = os.getenv('TFT_BUCKET', 'forecasting-paper')
REGION = os.getenv('AWS_REGION', 'eu-central-1')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', help='URI immagine ECR (obbligatorio per lanciare)')
    p.add_argument('--role', help='ARN del SageMaker execution role (obbligatorio per lanciare)')
    p.add_argument('--instance', default='ml.g4dn.2xlarge')
    p.add_argument('--input', default=f's3://{BUCKET}/tft-input/')
    p.add_argument('--output', default=f's3://{BUCKET}/tft-output/')
    p.add_argument('--checkpoints', default=f's3://{BUCKET}/tft-checkpoints/')
    p.add_argument('--subsample', default='0', help='SERIES_SUBSAMPLE (0 = tutte le serie)')
    p.add_argument('--hidden-cap', default='256')
    p.add_argument('--mode', default='all', choices=['all', 'hpo', 'cells'])
    p.add_argument('--max-hours', type=int, default=24)
    p.add_argument('--volume-gb', type=int, default=100)
    p.add_argument('--spot', action='store_true', help='managed spot training')
    p.add_argument('--status', help='mostra lo stato di un job esistente ed esci')
    a = p.parse_args()

    sm = boto3.client('sagemaker', region_name=REGION)

    if a.status:
        d = sm.describe_training_job(TrainingJobName=a.status)
        print(f"{d['TrainingJobName']}: {d['TrainingJobStatus']} ({d.get('SecondaryStatus')})")
        if d.get('FailureReason'):
            print('FailureReason:', d['FailureReason'])
        print('model artifacts:', d.get('ModelArtifacts', {}).get('S3ModelArtifacts'))
        return

    if not a.image or not a.role:
        sys.exit('Servono --image e --role. Vedi pipeline/sagemaker_README.md')

    job = f"tft-gpu-{time.strftime('%Y%m%d-%H%M%S')}"
    max_run = a.max_hours * 3600

    kw = dict(
        TrainingJobName=job,
        AlgorithmSpecification={'TrainingImage': a.image, 'TrainingInputMode': 'File'},
        RoleArn=a.role,
        InputDataConfig=[{
            'ChannelName': 'input',
            'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': a.input,
                'S3DataDistributionType': 'FullyReplicated',
            }},
            'InputMode': 'File',
        }],
        OutputDataConfig={'S3OutputPath': a.output},
        ResourceConfig={'InstanceType': a.instance, 'InstanceCount': 1,
                        'VolumeSizeInGB': a.volume_gb},
        CheckpointConfig={'S3Uri': a.checkpoints, 'LocalPath': '/opt/ml/checkpoints'},
        HyperParameters={
            'TFT_HIDDEN_CAP': a.hidden_cap,
            'TFT_PRECISION': '32-true',
            'TFT_MODE': a.mode,
            'SERIES_SUBSAMPLE': str(a.subsample),
        },
        StoppingCondition={'MaxRuntimeInSeconds': max_run},
    )
    if a.spot:
        kw['EnableManagedSpotTraining'] = True
        kw['StoppingCondition']['MaxWaitTimeInSeconds'] = max_run + 3600

    sm.create_training_job(**kw)
    print(f'Lanciato: {job}')
    print(f'  istanza   : {a.instance}{"  (SPOT)" if a.spot else ""}')
    print(f'  input     : {a.input}')
    print(f'  checkpoint: {a.checkpoints}   (resume automatico)')
    print(f'  output    : {a.output}{job}/output/model.tar.gz')
    print(f'\nStato:  python pipeline/sagemaker_launch.py --status {job}')
    print(f'Log  :  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job} --follow')


if __name__ == '__main__':
    main()
