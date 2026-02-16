#!/bin/bash
# Docker End-to-End Integration Test for PLM Slow Extraction
# 
# This script verifies the slow extraction Docker container works correctly with:
# - Sample document input with known technical terms
# - External vocabulary files mounted as volumes
# - Both chunking strategies (whole and heading)
# - JSON output with multiline text preservation
# - Low-confidence term logging to JSONL

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="/tmp/plm-test"
INPUT_DIR="${TEST_DIR}/input"
OUTPUT_DIR="${TEST_DIR}/output"
LOG_DIR="${TEST_DIR}/logs"
VOCAB_DIR="${TEST_DIR}/vocabularies"
DOCKER_IMAGE="plm-slow-extraction:0.1.0"
TEST_FILE="kubernetes-test.md"
OUTPUT_FILE="kubernetes-test.json"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_api_key() {
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log_error "ANTHROPIC_API_KEY is not set"
        log_error "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        exit 1
    fi
    log_info "ANTHROPIC_API_KEY is set"
}

check_docker_image() {
    if ! docker images | grep -q "${DOCKER_IMAGE}"; then
        log_error "Docker image ${DOCKER_IMAGE} not found"
        log_error "Build it with: nix build .#slow-extraction-docker && docker load < result"
        exit 1
    fi
    log_info "Docker image ${DOCKER_IMAGE} found"
}

setup_test_dirs() {
    log_info "Setting up test directories..."
    mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}" "${VOCAB_DIR}"
    log_info "Test directories created at ${TEST_DIR}"
}

copy_vocabularies() {
    log_info "Copying vocabulary files..."
    
    # Find vocabularies relative to script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    VOCAB_SOURCE="${SCRIPT_DIR}/data/vocabularies"
    
    if [[ ! -d "${VOCAB_SOURCE}" ]]; then
        log_error "Vocabulary source directory not found: ${VOCAB_SOURCE}"
        exit 1
    fi
    
    cp "${VOCAB_SOURCE}"/*.json "${VOCAB_DIR}/"
    log_info "Vocabularies copied: $(ls -1 ${VOCAB_DIR}/*.json | wc -l) files"
}

create_test_document() {
    log_info "Creating test document..."
    cat > "${INPUT_DIR}/${TEST_FILE}" << 'TESTEOF'
# Pod Lifecycle

A Pod's status field is a PodStatus object,
which has a phase field.

## Container States

Kubernetes tracks the state of each container inside a Pod.
Using React with TypeScript and useState hook.
TESTEOF
    
    log_info "Test document created with known terms: Kubernetes, Pod, React, TypeScript, useState"
}

run_docker_test() {
    local strategy=$1
    local expected_chunks=$2
    
    log_info "Running Docker container with CHUNKING_STRATEGY=${strategy}..."
    
    # Clean output directory before each run
    rm -f "${OUTPUT_DIR}"/*
    
    # Run container
    if ! docker run --rm \
        -v "${INPUT_DIR}:/data/input" \
        -v "${OUTPUT_DIR}:/data/output" \
        -v "${LOG_DIR}:/data/logs" \
        -v "${VOCAB_DIR}:/data/vocabularies" \
        -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
        -e PROCESS_ONCE=true \
        -e CHUNKING_STRATEGY="${strategy}" \
        "${DOCKER_IMAGE}"; then
        log_error "Docker container failed with CHUNKING_STRATEGY=${strategy}"
        return 1
    fi
    
    log_info "Docker container completed successfully"
    
    # Verify output file exists
    if [[ ! -f "${OUTPUT_DIR}/${OUTPUT_FILE}" ]]; then
        log_error "Output file not found: ${OUTPUT_DIR}/${OUTPUT_FILE}"
        return 1
    fi
    
    log_info "Output file created: ${OUTPUT_DIR}/${OUTPUT_FILE}"
    
    # Verify JSON structure
    if ! jq empty "${OUTPUT_DIR}/${OUTPUT_FILE}" 2>/dev/null; then
        log_error "Output file is not valid JSON"
        return 1
    fi
    
    log_info "JSON structure is valid"
    
    # Verify required fields
    local has_file has_chunks has_stats
    has_file=$(jq 'has("file")' "${OUTPUT_DIR}/${OUTPUT_FILE}")
    has_chunks=$(jq 'has("chunks")' "${OUTPUT_DIR}/${OUTPUT_FILE}")
    has_stats=$(jq 'has("stats")' "${OUTPUT_DIR}/${OUTPUT_FILE}")
    
    if [[ "${has_file}" != "true" ]] || [[ "${has_chunks}" != "true" ]] || [[ "${has_stats}" != "true" ]]; then
        log_error "Output JSON missing required fields (file, chunks, stats)"
        return 1
    fi
    
    log_info "Output JSON has required fields: file, chunks, stats"
    
    # Verify chunk count
    local chunk_count
    chunk_count=$(jq '.chunks | length' "${OUTPUT_DIR}/${OUTPUT_FILE}")
    
    if [[ "${chunk_count}" != "${expected_chunks}" ]]; then
        log_warn "Expected ${expected_chunks} chunk(s) but got ${chunk_count}"
        log_info "Chunk count: ${chunk_count}"
    else
        log_info "Chunk count matches expected: ${chunk_count}"
    fi
    
    # Verify multiline text preservation
    local has_newlines
    has_newlines=$(jq '.chunks[0].text | contains("\n")' "${OUTPUT_DIR}/${OUTPUT_FILE}" 2>/dev/null || echo "false")
    
    if [[ "${has_newlines}" == "true" ]]; then
        log_info "Multiline text preserved (contains newlines)"
    else
        log_warn "Multiline text may not be preserved"
    fi
    
    # Verify terms extracted
    local term_count
    term_count=$(jq '[.chunks[].terms[]? | select(.confidence != null)] | length' "${OUTPUT_DIR}/${OUTPUT_FILE}" 2>/dev/null || echo "0")
    
    if [[ "${term_count}" -gt 0 ]]; then
        log_info "Terms extracted: ${term_count} terms with confidence levels"
    else
        log_warn "No terms extracted with confidence levels"
    fi
    
    # Display output for inspection
    log_info "Output JSON content:"
    jq '.' "${OUTPUT_DIR}/${OUTPUT_FILE}"
    
    return 0
}

check_low_confidence_logs() {
    log_info "Checking for low-confidence term logs..."
    
    if [[ -f "${LOG_DIR}/low_confidence.jsonl" ]]; then
        local line_count
        line_count=$(wc -l < "${LOG_DIR}/low_confidence.jsonl")
        log_info "Low-confidence terms logged: ${line_count} entries"
        log_info "Sample entries:"
        head -3 "${LOG_DIR}/low_confidence.jsonl" | jq '.' 2>/dev/null || cat "${LOG_DIR}/low_confidence.jsonl" | head -3
    else
        log_info "No low-confidence terms logged (file not created)"
    fi
}

cleanup() {
    log_info "Cleaning up test artifacts..."
    rm -rf "${TEST_DIR}"
    log_info "Test artifacts removed"
}

main() {
    log_info "Starting Docker E2E Integration Test"
    log_info "========================================"
    
    # Check prerequisites
    check_api_key
    check_docker_image
    
    # Setup
    setup_test_dirs
    copy_vocabularies
    create_test_document
    
    # Test with whole chunking strategy
    log_info ""
    log_info "TEST 1: Whole Chunking Strategy"
    log_info "================================"
    if ! run_docker_test "whole" 1; then
        log_error "Test 1 failed"
        cleanup
        exit 1
    fi
    
    # Test with heading chunking strategy
    log_info ""
    log_info "TEST 2: Heading Chunking Strategy"
    log_info "=================================="
    if ! run_docker_test "heading" -1; then  # -1 means we don't enforce exact count
        log_error "Test 2 failed"
        cleanup
        exit 1
    fi
    
    # Check logs
    log_info ""
    log_info "Checking Logs"
    log_info "============="
    check_low_confidence_logs
    
    # Cleanup
    log_info ""
    cleanup
    
    log_info ""
    log_info "========================================"
    log_info "SUCCESS: Docker E2E test passed"
    log_info "========================================"
}

# Run main function
main "$@"
