#!/bin/bash

# Usage: ./condor.sh  your_job_name   ; set a unique job name

# Important commands 
#condor_status -compact       ; Check the cluster, Use with option -compact
#condor_q 		      ; Check the jobs runing on the queue
#condor_submit  job_script    ; Submit a job, See the bottom of this script
#condor_rm   job_ID	      ; Stop(cancel) and remove a sumitted job

outfile=$1
subfile=$outfile.sub

# make sure your Env ($PATH for cuda,  conda activate your_env, etc.)

cat > $subfile << EOF
Executable            = train.sh
Log                   = $outfile.log
Error                 = $outfile.err
Output                = $outfile.out
# NFS
+IwdFlusNFSCache      = False
Should_transfer_files = no
GetEnv                = True
# for ML jobs
Request_GPUs          = 1
# for TensorFlow-gpu
Requirements          = CUDACapability > 3.0
#Requirements          = machine == "node12.synapse"
# Prevent re-run
periodic_remove       = JobStatus == 1 && NumJobStarts > 0
# Email
Notification          = Always
Notify_user           = yunj1n_@pukyong.ac.kr
Queue
EOF

condor_submit $subfile

