# How-to Guides

Practical, task-oriented guides for common VibeML operations.

## Quick Navigation

- **[Launch Training Jobs](launch-jobs.md)** - Start training jobs with different configurations
- **[Track Budgets](track-budgets.md)** - Monitor spending and set budget limits
- **[Manage Credentials](manage-credentials.md)** - Securely store and manage cloud credentials
- **[Debug Cloud Issues](debug-cloud.md)** - Troubleshoot cloud provider problems
- **[Use MCP Server](use-mcp.md)** - Integrate with Claude Code and other MCP clients

## Guide Format

Each guide follows this structure:

1. **Prerequisites** - What you need before starting
2. **Steps** - Clear, numbered instructions
3. **Expected Results** - What success looks like
4. **Troubleshooting** - Common issues and solutions
5. **Next Steps** - Related guides and concepts

## Common Tasks

### Training Jobs

- Launch basic training job
- Use custom hyperparameters
- Multi-GPU training
- Resume from checkpoint
- Download trained models

### Cost Management

- Set budget limits
- View spending reports
- Compare pricing options
- Optimize GPU selection
- Use spot instances

### Configuration

- Store credentials securely
- Set default preferences
- Configure multiple clouds
- Manage budget alerts
- Custom workflow templates

### Monitoring & Debugging

- Check job status
- View training logs
- Debug cloud errors
- Handle quota limits
- Recover from failures

## Best Practices

Throughout these guides, we highlight best practices:

!!! tip "Best Practice"
    Always set a `max_cost` limit to avoid unexpected charges.

!!! warning "Common Mistake"
    Don't forget to terminate jobs when testing to avoid unnecessary costs.

!!! info "Pro Tip"
    Use spot instances for development and on-demand for production.

## Getting Help

If you can't find what you're looking for:

1. Check the [troubleshooting playbook](../playbooks/troubleshooting.md)
2. Review [common errors](../playbooks/common-errors.md)
3. Search the [API reference](../reference/index.md)
4. [Open an issue](https://github.com/prassanna-ravishankar/vibeml/issues) on GitHub
