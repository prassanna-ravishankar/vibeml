# Playbooks

Practical troubleshooting guides and recovery procedures.

## Contents

- **[Troubleshooting](troubleshooting.md)** - Systematic debugging approaches
- **[Common Errors](common-errors.md)** - Frequently encountered errors and solutions

## Quick Diagnostics

### Job Won't Launch

1. Check credentials: `vibeml config verify-credentials`
2. Verify model exists: Visit `https://huggingface.co/{model-id}`
3. Check budget limits: `vibeml config show`
4. Try with `--dry-run` flag
5. Check cloud provider status

### High Costs

1. Review GPU selection (use `--gpu-type RTX4090` for dev)
2. Enable spot instances (`--use-spot true`)
3. Reduce training steps (`--max-steps 100`)
4. Check current spending: `vibeml config show-spending`

### Slow Training

1. Verify GPU type is appropriate for model size
2. Check dataset size and format
3. Review hyperparameters (batch size, gradient accumulation)
4. Monitor GPU utilization: `nvidia-smi` on cluster

### Authentication Errors

1. Verify credentials: `vibeml config verify-credentials`
2. Check environment variables: `env | grep API_KEY`
3. Re-authenticate with cloud provider
4. Clear credential cache: `rm ~/.vibeml/credentials.enc`

## Error Categories

### Configuration Errors

- Missing credentials
- Invalid configuration values
- Permission issues
- File access problems

### Validation Errors

- Invalid model or dataset names
- Insufficient GPU memory
- Budget limit exceeded
- Incompatible hyperparameters

### Cloud Errors

- No GPU availability
- Quota limits reached
- Network connectivity issues
- Authentication failures

### Runtime Errors

- Training script failures
- Out of memory errors
- Dataset loading issues
- Checkpoint corruption

## Recovery Procedures

### Failed Job Recovery

```bash
# Check job status
vibeml get-job-status --cluster {cluster-name}

# SSH into cluster for debugging
sky ssh {cluster-name}

# View logs
tail -f ~/outputs/train.log

# Restart from checkpoint
vibeml launch-training ... --resume-from {checkpoint-path}
```

### Credential Reset

```bash
# Backup existing credentials
cp ~/.vibeml/credentials.enc ~/.vibeml/credentials.enc.bak

# Remove corrupted credentials
rm ~/.vibeml/credentials.enc ~/.vibeml/.key

# Reconfigure
vibeml config set-credential nebius api_key "your-key"
```

### Budget Reset

```bash
# View current spending
vibeml config show-spending

# Reset monthly spending (new billing cycle)
vibeml config reset-spending

# Adjust budget limits
vibeml config set-budget monthly 1000.0
```

## Best Practices

### Before Launching Jobs

- [ ] Verify credentials are configured
- [ ] Set appropriate budget limits
- [ ] Use `--dry-run` for testing
- [ ] Start with small `--max-steps` values
- [ ] Use development GPUs (RTX4090) for testing

### During Training

- [ ] Monitor job status regularly
- [ ] Check cost accumulation
- [ ] Review training logs for errors
- [ ] Verify GPU utilization
- [ ] Set up alerts for failures

### After Completion

- [ ] Download model artifacts promptly
- [ ] Terminate cluster to stop billing
- [ ] Review final costs
- [ ] Update budget tracking
- [ ] Document lessons learned

## Monitoring Checklist

```bash
# Check job status
vibeml get-job-status --cluster {name}

# View active jobs
vibeml list-jobs

# Check spending
vibeml config show-spending

# SSH to cluster for detailed logs
sky ssh {cluster-name}

# Inside cluster:
# - nvidia-smi (GPU utilization)
# - htop (CPU/memory)
# - tail -f ~/outputs/train.log
# - tensorboard --logdir ~/outputs
```

## Emergency Procedures

### Unexpected High Costs

```bash
# List all jobs
vibeml list-jobs

# Terminate expensive jobs
vibeml terminate-job --cluster {name}

# Verify termination
sky status

# Review final costs
vibeml config show-spending
```

### Cloud Provider Outage

1. Check provider status page
2. Try alternative region or cloud
3. Use `--cloud` flag to switch providers
4. Contact VibeML support if needed

### Data Loss Prevention

```bash
# Regular backups during long training
while true; do
  sky rsync {cluster}:~/outputs ./backups/$(date +%Y%m%d-%H%M%S)
  sleep 3600  # Every hour
done

# Checkpoint configuration
vibeml launch-training ... --checkpoint-steps 100
```

## Getting Help

If these playbooks don't resolve your issue:

1. Collect diagnostic information:
   ```bash
   vibeml debug-info > debug.txt
   ```

2. Check existing issues: [GitHub Issues](https://github.com/prassanna-ravishankar/vibeml/issues)

3. Create new issue with:
   - Debug information
   - Full error messages
   - Steps to reproduce
   - Environment details

4. Join community discussions

## Related Resources

- [Troubleshooting Guide](troubleshooting.md)
- [Common Errors Reference](common-errors.md)
- [How-to: Debug Cloud Issues](../how-to/debug-cloud.md)
- [API Reference: Exceptions](../reference/api/exceptions.md)
