for i in {0..41}
do
    start_idx=$((i * 50))
    end_idx=$(((i + 1) * 50))
    echo "Processing range $start_idx to $end_idx"
    # echo "python scrape_miner.py --start_idx=$start_idx --end_idx=$end_idx"
    sbatch -p 'deepaklong,all' --gres=gpu:1 --job-name='md_extract' --comment='md_extract' --time=6:00:00 -c8 --mem=50g --wrap="python scrape_miner.py --start_idx=$start_idx --end_idx=$end_idx" --output="slurm_logs/md_extract_${i}.out" --error="slurm_logs/md_extract_${i}.err"
done


