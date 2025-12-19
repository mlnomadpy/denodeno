#!/bin/bash
# Run all tests

echo "Running DenoMAE tests..."
python -m pytest tests/ -v --tb=short

echo "Tests complete!"
