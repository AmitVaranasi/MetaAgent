#!/bin/bash

################################################################################
# LinkedIn Outreach LaunchAgent Uninstaller
################################################################################
#
# Description:
#   Uninstalls and stops the LinkedIn outreach automation LaunchAgent.
#   Does NOT remove configuration files, logs, or user data.
#
# Usage:
#   ./uninstall_outreach.sh
#
# What this script does:
#   - Stops the LaunchAgent
#   - Removes the plist from ~/Library/LaunchAgents/
#   - Displays confirmation
#
# What this script does NOT do:
#   - Remove log files (preserved for your records)
#   - Remove configuration files
#   - Remove user data or database files
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
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_DEST="${LAUNCH_AGENTS_DIR}/${PLIST_FILENAME}"
LOGS_DIR="${HOME}/Library/Logs"

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

check_if_installed() {
    if [[ ! -f "${PLIST_DEST}" ]]; then
        print_warning "LaunchAgent is not installed at: ${PLIST_DEST}"
        return 1
    fi
    return 0
}

unload_agent() {
    print_info "Stopping LaunchAgent..."

    # Check if the agent is currently loaded
    if launchctl list | grep -q "${PLIST_LABEL}"; then
        if launchctl unload "${PLIST_DEST}" 2>&1; then
            print_success "LaunchAgent stopped successfully"
            sleep 1
        else
            print_warning "Failed to unload LaunchAgent (it may not be running)"
        fi
    else
        print_info "LaunchAgent is not currently running"
    fi
}

remove_plist() {
    print_info "Removing plist file..."

    if [[ -f "${PLIST_DEST}" ]]; then
        rm "${PLIST_DEST}"
        print_success "Removed: ${PLIST_DEST}"
    else
        print_warning "Plist file not found (may already be removed)"
    fi
}

verify_uninstallation() {
    print_info "Verifying uninstallation..."

    local still_running=false

    if launchctl list | grep -q "${PLIST_LABEL}"; then
        print_error "LaunchAgent is still running"
        still_running=true
    fi

    if [[ -f "${PLIST_DEST}" ]]; then
        print_error "Plist file still exists"
        still_running=true
    fi

    if [[ "$still_running" == "false" ]]; then
        print_success "LaunchAgent successfully uninstalled"
        return 0
    else
        print_error "Uninstallation incomplete"
        return 1
    fi
}

show_preserved_files() {
    echo ""
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║              Preserved Files and Data                  ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}The following files were NOT removed:${NC}"
    echo ""

    # Check for log files
    if [[ -f "${LOGS_DIR}/meta-agent-linkedin-outreach.log" ]]; then
        echo "  • ${LOGS_DIR}/meta-agent-linkedin-outreach.log"
    fi

    if [[ -f "${LOGS_DIR}/meta-agent-linkedin-outreach-error.log" ]]; then
        echo "  • ${LOGS_DIR}/meta-agent-linkedin-outreach-error.log"
    fi

    echo "  • Configuration files in project directory"
    echo "  • User data and database files"
    echo ""
    echo -e "${BLUE}To remove log files manually:${NC}"
    echo "  rm ~/Library/Logs/meta-agent-linkedin-outreach*.log"
    echo ""
}

show_post_uninstall_info() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║       Uninstallation Completed Successfully!          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}The LinkedIn Outreach LaunchAgent has been removed.${NC}"
    echo ""
    echo -e "${BLUE}To reinstall:${NC}"
    echo "  ./install_outreach.sh"
    echo ""
}

################################################################################
# Main Uninstallation Process
################################################################################

main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    LinkedIn Outreach LaunchAgent Uninstaller          ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Pre-uninstallation checks
    check_macos

    if ! check_if_installed; then
        print_info "Nothing to uninstall."
        exit 0
    fi

    # Confirm with user
    echo -e "${YELLOW}This will stop and remove the LinkedIn Outreach LaunchAgent.${NC}"
    echo -e "${YELLOW}Your configuration files and logs will be preserved.${NC}"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Uninstallation cancelled."
        exit 0
    fi

    echo ""

    # Perform uninstallation
    unload_agent
    remove_plist

    # Verify and show results
    if verify_uninstallation; then
        show_post_uninstall_info
        show_preserved_files
    else
        print_error "Uninstallation encountered errors. Please check manually."
        exit 1
    fi
}

# Run main function
main "$@"
