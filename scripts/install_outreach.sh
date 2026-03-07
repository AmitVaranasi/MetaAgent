#!/bin/bash

################################################################################
# LinkedIn Outreach LaunchAgent Installer
################################################################################
#
# Description:
#   Installs and starts the LinkedIn outreach automation as a macOS LaunchAgent.
#   The agent will run automatically on login and restart if it crashes.
#
# Usage:
#   ./install_outreach.sh
#
# Requirements:
#   - macOS
#   - Python 3 installed
#   - meta_agent package available in project
#
# Installation Location:
#   ~/Library/LaunchAgents/com.meta-agent.linkedin-outreach.plist
#
# Log Files:
#   ~/Library/Logs/meta-agent-linkedin-outreach.log
#   ~/Library/Logs/meta-agent-linkedin-outreach-error.log
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PLIST_LABEL="com.meta-agent.linkedin-outreach"
PLIST_FILENAME="${PLIST_LABEL}.plist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
LOGS_DIR="${HOME}/Library/Logs"
PLIST_SOURCE="${SCRIPT_DIR}/${PLIST_FILENAME}"
PLIST_DEST="${LAUNCH_AGENTS_DIR}/${PLIST_FILENAME}"

################################################################################
# Helper Functions
################################################################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        print_error "This script is for macOS only."
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3."
        exit 1
    fi

    local python_path
    python_path="$(which python3)"
    print_info "Found Python 3 at: ${python_path}"
    echo "$python_path"
}

check_project_structure() {
    if [[ ! -d "${PROJECT_DIR}/meta_agent" ]]; then
        print_error "meta_agent package not found at ${PROJECT_DIR}/meta_agent"
        exit 1
    fi
    print_info "Project structure verified"
}

unload_existing_agent() {
    if launchctl list | grep -q "${PLIST_LABEL}"; then
        print_info "Unloading existing agent..."
        launchctl unload "${PLIST_DEST}" 2>/dev/null || true
        sleep 1
    fi
}

create_launch_agents_dir() {
    if [[ ! -d "${LAUNCH_AGENTS_DIR}" ]]; then
        print_info "Creating LaunchAgents directory..."
        mkdir -p "${LAUNCH_AGENTS_DIR}"
    fi
}

create_logs_dir() {
    if [[ ! -d "${LOGS_DIR}" ]]; then
        print_info "Creating Logs directory..."
        mkdir -p "${LOGS_DIR}"
    fi
}

update_plist_paths() {
    local python_path="$1"
    local temp_plist
    temp_plist=$(mktemp)

    print_info "Configuring plist with project paths..."

    # Replace placeholders in the plist
    sed -e "s|PYTHON_PATH_PLACEHOLDER|${python_path}|g" \
        -e "s|PROJECT_PATH_PLACEHOLDER|${PROJECT_DIR}|g" \
        -e "s|LOG_PATH_PLACEHOLDER|${LOGS_DIR}|g" \
        -e "s|ENV_PATH_PLACEHOLDER|${PATH}|g" \
        "${PLIST_SOURCE}" > "$temp_plist"

    # Copy to destination
    cp "$temp_plist" "${PLIST_DEST}"
    rm "$temp_plist"

    print_success "Plist configured at: ${PLIST_DEST}"
}

load_agent() {
    print_info "Loading LaunchAgent..."

    if launchctl load "${PLIST_DEST}" 2>&1; then
        print_success "LaunchAgent loaded successfully"
        return 0
    else
        print_error "Failed to load LaunchAgent"
        return 1
    fi
}

verify_installation() {
    print_info "Verifying installation..."

    if launchctl list | grep -q "${PLIST_LABEL}"; then
        print_success "LaunchAgent is running"

        # Show the PID if available
        local status
        status=$(launchctl list | grep "${PLIST_LABEL}")
        echo -e "${GREEN}Status:${NC} ${status}"
        return 0
    else
        print_warning "LaunchAgent is not running"
        return 1
    fi
}

show_post_install_info() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        Installation Completed Successfully!           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "  • Agent Label:    ${PLIST_LABEL}"
    echo "  • Project Dir:    ${PROJECT_DIR}"
    echo "  • Plist Location: ${PLIST_DEST}"
    echo ""
    echo -e "${BLUE}Log Files:${NC}"
    echo "  • Standard Output: ${LOGS_DIR}/meta-agent-linkedin-outreach.log"
    echo "  • Error Output:    ${LOGS_DIR}/meta-agent-linkedin-outreach-error.log"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  • Check status:    ./check_outreach_status.sh"
    echo "  • View logs:       tail -f ~/Library/Logs/meta-agent-linkedin-outreach.log"
    echo "  • Uninstall:       ./uninstall_outreach.sh"
    echo "  • Restart agent:   launchctl kickstart -k gui/\$(id -u)/${PLIST_LABEL}"
    echo ""
}

################################################################################
# Main Installation Process
################################################################################

main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    LinkedIn Outreach LaunchAgent Installer            ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Pre-installation checks
    check_macos
    check_project_structure
    local python_path
    python_path=$(check_python)

    # Create necessary directories
    create_launch_agents_dir
    create_logs_dir

    # Unload existing agent if running
    unload_existing_agent

    # Configure and install plist
    update_plist_paths "$python_path"

    # Load the agent
    if load_agent; then
        sleep 2
        verify_installation
        show_post_install_info
    else
        print_error "Installation failed. Check the logs for details:"
        echo "  tail ${LOGS_DIR}/meta-agent-linkedin-outreach-error.log"
        exit 1
    fi
}

# Run main function
main "$@"
