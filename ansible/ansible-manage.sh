#!/bin/bash
# Ansible management script for crypto-forecasting project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSIBLE_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$ANSIBLE_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if ansible is installed
check_ansible() {
    if ! command -v ansible &> /dev/null; then
        print_error "Ansible is not installed. Please install it first:"
        echo "  pip install ansible"
        exit 1
    fi

    print_status "Ansible version: $(ansible --version | head -n1)"
}

# Show usage information
show_help() {
    cat << EOF
Crypto Forecasting Ansible Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    # Infrastructure Management
    setup                       Initial setup of the environment
    deploy [ENV]               Deploy the application (ENV: development, staging, production)
    infrastructure             Setup infrastructure (dependencies, Docker, K8s)
    
    # Database and Cache Management
    database                   Setup and configure PostgreSQL database
    redis                      Setup and configure Redis cache
    
    # Monitoring and Security
    monitoring                 Setup Prometheus and Grafana monitoring
    security                   Apply security hardening configuration
    performance                Optimize performance settings
    
    # Health and Recovery
    health                     Run comprehensive health checks
    health-monitor [action]    Control health monitoring daemon (start|stop|restart|status)
    backup                     Create backup of data and database
    restore [FILE]             Restore from backup file
    disaster-recovery          Run disaster recovery procedures
    test-dr                    Test disaster recovery procedures
    
    # Application Management
    start                      Start application services
    stop                       Stop application services
    restart                    Restart application services
    update                     Update application from git and restart
    status                     Show application status
    logs [SERVICE]             Show logs for a service

Options:
    -h, --help         Show this help message
    -v, --verbose      Enable verbose output
    -e KEY=VALUE       Set extra variables

Examples:
    $0 setup                    # Initial environment setup
    $0 infrastructure           # Setup infrastructure
    $0 deploy development       # Deploy to development
    $0 database                 # Setup database
    $0 monitoring               # Setup monitoring
    $0 health                   # Run health checks
    $0 health-monitor start     # Start health monitoring
    $0 backup                   # Create backup
    $0 restore backup_file.tar  # Restore from backup
    $0 disaster-recovery        # Run disaster recovery
    $0 test-dr                  # Test disaster recovery
    $0 logs backend             # Show backend logs

EOF
}

# Run ansible playbook with common options
run_playbook() {
    local playbook="$1"
    shift
    local extra_args="$@"

    print_status "Running playbook: $playbook"

    cd "$ANSIBLE_DIR"
    ansible-playbook \
        -i inventory/hosts.yml \
        "playbooks/$playbook" \
        $extra_args
}

# Main command dispatcher
main() {
    cd "$ANSIBLE_DIR"

    # Check prerequisites
    check_ansible

    case "${1:-help}" in
        # Infrastructure Management
        setup)
            print_header "Setting up environment"
            run_playbook "infrastructure-setup.yml" "${@:2}"
            ;;

        deploy)
            local env="${2:-development}"
            print_header "Deploying crypto-forecasting application to $env"
            run_playbook "main-deployment.yml" -e "env=$env" "${@:3}"
            ;;

        infrastructure)
            print_header "Setting up infrastructure"
            run_playbook "infrastructure-setup.yml" "${@:2}"
            ;;

        # Database and Cache Management
        database)
            print_header "Setting up database"
            run_playbook "database-setup.yml" "${@:2}"
            ;;

        redis)
            print_header "Setting up Redis cache"
            run_playbook "redis-setup.yml" "${@:2}"
            ;;

        # Monitoring and Security
        monitoring)
            print_header "Setting up monitoring"
            run_playbook "monitoring-setup.yml" "${@:2}"
            ;;

        security)
            print_header "Applying security hardening"
            run_playbook "security-hardening.yml" "${@:2}"
            ;;

        performance)
            print_header "Optimizing performance"
            run_playbook "performance-optimization.yml" "${@:2}"
            ;;

        # Health and Recovery
        health)
            print_header "Running health checks"
            run_playbook "health-check.yml" "${@:2}"
            ;;

        health-monitor)
            local action="${2:-status}"
            print_header "Health monitor: $action"
            case "$action" in
                start|stop|restart|status)
                    /opt/crypto-forecasting/health-monitor.sh "$action"
                    ;;
                *)
                    print_error "Invalid health-monitor action: $action"
                    print_status "Valid actions: start, stop, restart, status"
                    exit 1
                    ;;
            esac
            ;;

        backup)
            print_header "Creating backup"
            run_playbook "backup.yml" "${@:2}"
            ;;

        restore)
            local backup_file="${2}"
            if [[ -z "$backup_file" ]]; then
                print_error "Please specify backup file to restore"
                exit 1
            fi
            print_header "Restoring from backup: $backup_file"
            run_playbook "backup.yml" -e "operation=restore" -e "backup_file=$backup_file" "${@:3}"
            ;;

        disaster-recovery)
            print_header "Running disaster recovery procedures"
            run_playbook "disaster-recovery.yml" "${@:2}"
            ;;

        test-dr)
            print_header "Testing disaster recovery"
            if [[ -f "/opt/crypto-forecasting/disaster-recovery/scripts/test-dr.sh" ]]; then
                /opt/crypto-forecasting/disaster-recovery/scripts/test-dr.sh
            else
                print_error "Disaster recovery test script not found. Run 'disaster-recovery' first."
                exit 1
            fi
            ;;

        # Application Management
        start)
            print_header "Starting application services"
            run_playbook "application-deployment.yml" -e "operation=start" "${@:2}"
            ;;

        stop)
            print_header "Stopping application services"
            run_playbook "application-deployment.yml" -e "operation=stop" "${@:2}"
            ;;

        restart)
            print_header "Restarting application services"
            run_playbook "application-deployment.yml" -e "operation=restart" "${@:2}"
            ;;

        update)
            print_header "Updating application"
            run_playbook "application-deployment.yml" -e "operation=update" "${@:2}"
            ;;

        status)
            print_header "Checking application status"
            print_status "Running health checks..."
            if /opt/crypto-forecasting/health-check.sh; then
                print_status "✅ All systems operational"
            else
                print_warning "⚠️ Some issues detected - check logs for details"
            fi
            ;;

        logs)
            local service="${2:-backend}"
            print_header "Showing logs for service: $service"
            if command -v kubectl &> /dev/null; then
                kubectl logs -n crypto-forecasting deployment/"$service" -f
            else
                docker-compose -f "$PROJECT_ROOT/docker-compose.yml" logs -f "$service"
            fi
            ;;

        help|--help|-h)
            show_help
            ;;

        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
