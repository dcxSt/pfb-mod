#!/bin/bash
#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:40:00
#SBATCH --job-name snr
#SBATCH --output=snr_%j.txt
#SBATCH --mail-user=stephen.fay@mail.mcgill.ca
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.8

# Execute 32 jobs in parallel
(echo "starting job 01" && python3.8 snr_analysis_corr.py 1005 1 && echo "job 01 finished") &
(echo "starting job 02" && python3.8 snr_analysis_corr.py 1006 1 && echo "job 02 finished") &
(echo "starting job 03" && python3.8 snr_analysis_corr.py 1007 1 && echo "job 03 finished") &
(echo "starting job 04" && python3.8 snr_analysis_corr.py 1008 1 && echo "job 04 finished") &
(echo "starting job 05" && python3.8 snr_analysis_corr.py 1009 1 && echo "job 05 finished") &
(echo "starting job 06" && python3.8 snr_analysis_corr.py 1010 1 && echo "job 06 finished") &
(echo "starting job 07" && python3.8 snr_analysis_corr.py 1011 1 && echo "job 07 finished") &
(echo "starting job 08" && python3.8 snr_analysis_corr.py 1012 1 && echo "job 08 finished") &

(echo "starting job 09" && python3.8 snr_analysis_corr.py 1013 1 && echo "job 09 finished") &
(echo "starting job 10" && python3.8 snr_analysis_corr.py 1014 1 && echo "job 10 finished") &
(echo "starting job 11" && python3.8 snr_analysis_corr.py 1015 1 && echo "job 11 finished") &
(echo "starting job 12" && python3.8 snr_analysis_corr.py 1016 1 && echo "job 12 finished") &
(echo "starting job 13" && python3.8 snr_analysis_corr.py 1017 1 && echo "job 13 finished") &
(echo "starting job 14" && python3.8 snr_analysis_corr.py 1018 1 && echo "job 14 finished") &
(echo "starting job 15" && python3.8 snr_analysis_corr.py 1019 1 && echo "job 15 finished") &
(echo "starting job 16" && python3.8 snr_analysis_corr.py 1020 1 && echo "job 16 finished") &

(echo "starting job 17" && python3.8 snr_analysis_corr.py 1021 1 && echo "job 17 finished") &
(echo "starting job 18" && python3.8 snr_analysis_corr.py 1022 1 && echo "job 18 finished") &
(echo "starting job 19" && python3.8 snr_analysis_corr.py 1023 1 && echo "job 19 finished") &
(echo "starting job 20" && python3.8 snr_analysis_corr.py 1024 1 && echo "job 20 finished") &
(echo "starting job 21" && python3.8 snr_analysis_corr.py 1025 1 && echo "job 21 finished") &
(echo "starting job 22" && python3.8 snr_analysis_corr.py 1026 1 && echo "job 22 finished") &
(echo "starting job 23" && python3.8 snr_analysis_corr.py 1027 1 && echo "job 23 finished") &
(echo "starting job 24" && python3.8 snr_analysis_corr.py 1028 1 && echo "job 24 finished") &

(echo "starting job 25" && python3.8 snr_analysis_corr.py 1029 1 && echo "job 25 finished") &
(echo "starting job 26" && python3.8 snr_analysis_corr.py 1030 1 && echo "job 26 finished") &
(echo "starting job 27" && python3.8 snr_analysis_corr.py 1031 1 && echo "job 27 finished") &
(echo "starting job 28" && python3.8 snr_analysis_corr.py 1032 1 && echo "job 28 finished") &
(echo "starting job 29" && python3.8 snr_analysis_corr.py 1033 1 && echo "job 29 finished") &
(echo "starting job 30" && python3.8 snr_analysis_corr.py 1034 1 && echo "job 30 finished") &
(echo "starting job 31" && python3.8 snr_analysis_corr.py 1035 1 && echo "job 31 finished") &
(echo "starting job 32" && python3.8 snr_analysis_corr.py 1036 1 && echo "job 32 finished") &
wait

