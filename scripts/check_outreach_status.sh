#!/bin/bash

################################################################################
# LinkedIn Outreach LaunchAgent Status Checker
################################################################################
#
# Description:
#   Displays the current status of the LinkedIn outreach automation LaunchAgent.
#   Shows if the agent is running, recent log entries, and configuration info.
#
# Usage:
#   ./check_outreach_status.sh [OPTIONS]
#
# Options:
#   -v, --verbose    Show more detailed information
#   -l, --logs N     Show last N lines of logs (default: 20)
#   -h, --help       Show this help message
#
# Examples:
#   ./check_outreach_status.sh              # Basic status
#   ./check_outreach_status.sh -v           # Verbose status
#   ./check_outreach_status.sh --logs 50    # Show last 50 log lines
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PLIST_LABEL="com.meta-agent.linkedin-outreach"
PLIST_FILENAME="${PLIST_LABEL}.plist"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_DEST="${LAUNCH_AGENTS_DIR}/${PLIST_FILENAME}"
LOGS_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOGS_DIR}/meta-agent-linkedin-outreach.log"
ERROR_LOG_FILE="${LOGS_DIR}/meta-agent-linkedin-outreach-error.log"

# Options
VERBOSE=false
LOG_LINES=20

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '═%.0s' $(seq 1 ${#1}))${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_field() {
    local label="$1"
    local value="$2"
    printf "  ${CYAN}%-20s${NC} %s\n" "$label:" "$value"
}

show_help() {
    sed -n '/^##/,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -l|--logs)
                LOG_LINES="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

check_agent_status() {
    print_header "LaunchAgent Status"

    # Check if plist exists
    if [[ ! -f "${PLIST_DEST}" ]]; then
        print_error "LaunchAgent is NOT installed"
        print_field "Expected location" "${PLIST_DEST}"
        echo ""
        echo -e "${YELLOW}To install, run:${NC} ./install_outreach.sh"
        return 1
    fi

    print_success "LaunchAgent is installed"
    print_field "Plist location" "${PLIST_DEST}"

    # Check if agent is loaded
    if launchctl list | grep -q "${PLIST_LABEL}"; then
        local status_line
        status_line=$(launchctl list | grep "${PLIST_LABEL}")

        # Parse the status line (PID, Status, Label)
        local pid status
        pid=$(echo "$status_line" | awk '{print $1}')
        status=$(echo "$status_line" | awk '{print $2}')

        if [[ "$pid" != "-" ]]; then
            print_success "LaunchAgent is RUNNING"
            print_field "Process ID (PID)" "$pid"
            print_field "Exit status" "$status"

            # Show process details if verbose
            if [[ "$VERBOSE" == "true" ]]; then
                echo ""
                print_header "Process Details"
                ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,command 2>/dev/null || true
            fi
        else
            print_warning "LaunchAgent is loaded but NOT running"
            print_field "Last exit status" "$status"
        fi
    else
        print_error "LaunchAgent is NOT loaded"
        echo ""
        echo -e "${YELLOW}To start, run:${NC} launchctl load ${PLIST_DEST}"
    fi

    echo ""
}

show_plist_config() {
    if [[ "$VERBOSE" == "true" && -f "${PLIST_DEST}" ]]; then
        print_header "LaunchAgent Configuration"

        # Extract key configuration values
        local working_dir python_cmd
        working_dir=$(defaults read "${PLIST_DEST%.plist}" WorkingDirectory 2>/dev/null || echo "N/A")
        python_cmd=$(defaults read "${PLIST_DEST%.plist}" ProgramArguments 2>/dev/null | head -2 | tail -1 | xargs || echo "N/A")

        print_field "Working Directory" "$working_dir"
        print_field "Python Command" "$python_cmd"
        print_field "Run at Login" "$(defaults read "${PLIST_DEST%.plist}" RunAtLoad 2>/dev/null || echo "N/A")"
        print_field "Keep Alive" "$(defaults read "${PLIST_DEST%.plist}" KeepAlive 2>/dev/null || echo "N/A")"

        echo ""
    fi
}

show_log_files() {
    print_header "Log Files"

    # Standard output log
    if [[ -f "${LOG_FILE}" ]]; then
        local log_size log_modified
        log_size=$(du -h "${LOG_FILE}" | cut -f1)
        log_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${LOG_FILE}")

        print_success "Standard output log exists"
        print_field "Location" "${LOG_FILE}"
        print_field "Size" "$log_size"
        print_field "Last modified" "$log_modified"
    else
        print_warning "Standard output log not found"
        print_field "Expected location" "${LOG_FILE}"
    fi

    echo ""

    # Error log
    if [[ -f "${ERROR_LOG_FILE}" ]]; then
        local err_size err_modified
        err_size=$(du -h "${ERROR_LOG_FILE}" | cut -f1)
        err_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${ERROR_LOG_FILE}")

        # Check if there are recent errors
        local err_lines
        err_lines=$(wc -l < "${ERROR_LOG_FILE}" | xargs)

        if [[ "$err_lines" -gt 0 ]]; then
            print_warning "Error log exists with ${err_lines} lines"
        else
            print_success "Error log exists (empty)"
        fi

        print_field "Location" "${ERROR_LOG_FILE}"
        print_field "Size" "$err_size"
        print_field "Last modified" "$err_modified"
    else
        print_info "Error log not found (normal if no errors)"
        print_field "Expected location" "${ERROR_LOG_FILE}"
    fi

    echo ""
}

show_recent_logs() {
    print_header "Recent Log Entries (last ${LOG_LINES} lines)"

    if [[ -f "${LOG_FILE}" ]]; then
        echo -e "${CYAN}Standard Output:${NC}"
        tail -n "$LOG_LINES" "${LOG_FILE}" 2>/dev/null | sed 's/^/  /' || echo "  (empty)"
        echo ""
    fi

    if [[ -f "${ERROR_LOG_FILE}" ]] && [[ -s "${ERROR_LOG_FILE}" ]]; then
        echo -e "${YELLOW}Error Output:${NC}"
        tail -n "$LOG_LINES" "${ERROR_LOG_FILE}" 2>/dev/null | sed 's/^/  /' || echo "  (empty)"
        echo ""
    fi
}

show_last_run_time() {
    print_header "Last Activity"

    if [[ -f "${LOG_FILE}" ]]; then
        local last_line last_timestamp
        last_line=$(tail -n 1 "${LOG_FILE}" 2>/dev/null || echo "")

        if [[ -n "$last_line" ]]; then
            # Try to extract timestamp if log uses standard format
            print_field "Last log entry" "$(echo "$last_line" | cut -c1-80)..."

            local log_time
            log_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${LOG_FILE}")
            print_field "Log last updated" "$log_time"
        else
            print_info "No log entries yet"
        fi
    else
        print_info "No log file found"
    fi

    echo ""
}

show_useful_commands() {
    print_header "Useful Commands"

    echo -e "${CYAN}View live logs:${NC}"
    echo "  tail -f ${LOG_FILE}"
    echo ""

    echo -e "${CYAN}Restart agent:${NC}"
    echo "  launchctl kickstart -k gui/\$(id -u)/${PLIST_LABEL}"
    echo ""

    echo -e "${CYAN}Stop agent:${NC}"
    echo "  launchctl unload ${PLIST_DEST}"
    echo ""

    echo -e "${CYAN}Start agent:${NC}"
    echo "  launchctl load ${PLIST_DEST}"
    echo ""

    echo -e "${CYAN}Uninstall:${NC}"
    echo "  ./uninstall_outreach.sh"
    echo ""
}

################################################################################
# Main Status Check
################################################################################

main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    LinkedIn Outreach LaunchAgent Status               ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""

    parse_args "$@"

    check_agent_status
    show_plist_config
    show_log_files
    show_recent_logs
    show_last_run_time
    show_useful_commands
}

# Run main function
main "$@"
