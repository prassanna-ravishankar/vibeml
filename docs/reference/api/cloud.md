# Cloud Integration

Cost estimation and cloud provider configuration.

## CostEstimator

::: vibeml.cloud.CostEstimator
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      show_signature_annotations: true

## GPUPricing

::: vibeml.cloud.cost_estimator.GPUPricing
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Cost Estimation

```python
from vibeml.cloud import CostEstimator

estimator = CostEstimator()

# Estimate cost for a training job
cost = estimator.estimate_cost(
    gpu_type="L40S",
    estimated_hours=12.0,
    gpu_count=1,
    use_spot=True,
    cloud="nebius"
)

print(f"Hourly rate: ${cost.hourly_rate:.2f}/hr")
print(f"Expected cost: ${cost.expected_cost:.2f}")
print(f"Range: ${cost.min_cost:.2f} - ${cost.max_cost:.2f}")
```

### Duration Estimation

```python
# Estimate training duration
duration = estimator.estimate_duration(
    model_size="7b",
    dataset_rows=10000,
    max_steps=1000
)

print(f"Estimated duration: {duration:.1f} hours")
```

### Multi-GPU Cost Calculation

```python
# Calculate cost for multi-GPU training
cost_8gpu = estimator.estimate_cost(
    gpu_type="H100",
    estimated_hours=8.0,
    gpu_count=8,  # 8x H100
    use_spot=False,
    cloud="nebius"
)

print(f"8x H100 cost: ${cost_8gpu.expected_cost:.2f}")
```

### Spot vs On-Demand Comparison

```python
# Compare spot and on-demand pricing
on_demand = estimator.estimate_cost(
    gpu_type="L40S",
    estimated_hours=10.0,
    use_spot=False
)

spot = estimator.estimate_cost(
    gpu_type="L40S",
    estimated_hours=10.0,
    use_spot=True
)

savings = on_demand.expected_cost - spot.expected_cost
savings_pct = (savings / on_demand.expected_cost) * 100

print(f"On-demand: ${on_demand.expected_cost:.2f}")
print(f"Spot: ${spot.expected_cost:.2f}")
print(f"Savings: ${savings:.2f} ({savings_pct:.1f}%)")
```

## GPU Pricing

Current Nebius GPU pricing (per hour, on-demand):

| GPU Type | Memory | Hourly Rate | Spot Discount |
|----------|--------|-------------|---------------|
| RTX4090  | 24GB   | $0.80       | 70% (30% of on-demand) |
| L40S     | 48GB   | $1.20       | 70% |
| A100     | 80GB   | $2.40       | 70% |
| H100     | 80GB   | $3.50       | 70% |

### Custom Pricing

Override default pricing for custom cloud providers:

```python
from vibeml.cloud.cost_estimator import GPUPricing

estimator = CostEstimator()

# Add custom GPU pricing
estimator.NEBIUS_PRICING["CustomGPU"] = GPUPricing(
    hourly_rate=5.0,
    spot_discount=0.6  # 40% of on-demand
)

cost = estimator.estimate_cost(
    gpu_type="CustomGPU",
    estimated_hours=10.0
)
```

## Cost Variance

The estimator accounts for training time variance:

- **Minimum cost**: 80% of expected duration
- **Expected cost**: 100% of estimated duration
- **Maximum cost**: 130% of estimated duration

This provides a realistic cost range accounting for:
- Dataset processing variability
- Model convergence speed
- Infrastructure overhead
- Potential interruptions (spot instances)

## See Also

- [Data Models](models.md) - CostEstimate model documentation
- [Validation](validation.md) - Resource validation and GPU selection
- [Cost Management Concept](../../concepts/cost-management.md)
