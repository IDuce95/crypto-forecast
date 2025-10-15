

echo "=== Starting Airflow for Crypto Forecasting ==="

export AIRFLOW_HOME="/home/palianm/Desktop/crypto-forecasting/airflow"
export PYTHONPATH="/home/palianm/Desktop/crypto-forecasting:$PYTHONPATH"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0
    else
        return 1
    fi
}

wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if eval $check_command > /dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

setup_environment() {
    print_header "Setting up Airflow Environment"
    
    mkdir -p "$AIRFLOW_HOME"/{dags,plugins,logs,config}
    
    cd /home/palianm/Desktop/crypto-forecasting
    python airflow/airflow_setup.py
    
    print_status "Environment setup completed"
}

init_database() {
    print_header "Initializing Airflow Database"
    
    if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
        print_status "Initializing Airflow database..."
        airflow db init
        
        print_status "Creating admin user..."
        airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@crypto-forecasting.com \
            --password admin123
    else
        print_status "Database already exists, upgrading..."
        airflow db upgrade
    fi
}

set_variables() {
    print_header "Setting Airflow Variables"
    
    airflow variables set crypto_symbols "BTC,ETH,ADA,DOT,LINK"
    airflow variables set data_start_date "2023-01-01"
    airflow variables set accuracy_threshold "0.75"
    airflow variables set confidence_threshold "0.80"
    airflow variables set ml_model_types "random_forest,xgboost,lgbm"
    airflow variables set prediction_horizon_days "7"
    airflow variables set cache_ttl_seconds "3600"
    airflow variables set notification_email "admin@crypto-forecasting.com"
    
    print_status "Variables set successfully"
}

start_webserver() {
    print_header "Starting Airflow Webserver"
    
    if check_process "airflow webserver"; then
        print_warning "Webserver is already running"
        return 0
    fi
    
    print_status "Starting webserver on port 8080..."
    nohup airflow webserver --port 8080 > "$AIRFLOW_HOME/logs/webserver.log" 2>&1 &
    
    wait_for_service "Airflow Webserver" "curl -s http://localhost:8080/health"
}

start_scheduler() {
    print_header "Starting Airflow Scheduler"
    
    if check_process "airflow scheduler"; then
        print_warning "Scheduler is already running"
        return 0
    fi
    
    print_status "Starting scheduler..."
    nohup airflow scheduler > "$AIRFLOW_HOME/logs/scheduler.log" 2>&1 &
    
    wait_for_service "Airflow Scheduler" "airflow dags list"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    if ! redis-cli ping > /dev/null 2>&1; then
        print_warning "Redis is not running. Starting Redis..."
        if command -v redis-server > /dev/null; then
            nohup redis-server > /dev/null 2>&1 &
            sleep 2
            if redis-cli ping > /dev/null 2>&1; then
                print_status "Redis started successfully"
            else
                print_error "Failed to start Redis"
            fi
        else
            print_error "Redis is not installed"
        fi
    else
        print_status "Redis is running"
    fi
    
    if [ ! -f "/home/palianm/Desktop/crypto-forecasting/venv/bin/python" ]; then
        print_error "Virtual environment not found"
        exit 1
    else
        print_status "Python virtual environment found"
    fi
    
    source /home/palianm/Desktop/crypto-forecasting/venv/bin/activate
    print_status "Virtual environment activated"
    
    python -c "import airflow" 2>/dev/null || {
        print_error "Airflow not installed in virtual environment"
        exit 1
    }
    print_status "Airflow package found"
}

stop_services() {
    print_header "Stopping Airflow Services"
    
    if check_process "airflow webserver"; then
        print_status "Stopping webserver..."
        pkill -f "airflow webserver"
    fi
    
    if check_process "airflow scheduler"; then
        print_status "Stopping scheduler..."
        pkill -f "airflow scheduler"
    fi
    
    print_status "All services stopped"
}

show_status() {
    print_header "Airflow Services Status"
    
    if check_process "airflow webserver"; then
        print_status "Webserver: RUNNING"
    else
        print_warning "Webserver: STOPPED"
    fi
    
    if check_process "airflow scheduler"; then
        print_status "Scheduler: RUNNING"
    else
        print_warning "Scheduler: STOPPED"
    fi
    
    if redis-cli ping > /dev/null 2>&1; then
        print_status "Redis: RUNNING"
    else
        print_warning "Redis: STOPPED"
    fi
    
    echo ""
    print_status "Airflow UI: http://localhost:8080"
    print_status "Login: admin / admin123"
}

restart_services() {
    print_header "Restarting Airflow Services"
    stop_services
    sleep 3
    start_all
}

start_all() {
    check_prerequisites
    setup_environment
    init_database
    set_variables
    start_webserver
    start_scheduler
    show_status
}

show_logs() {
    local service=$1
    case $service in
        "webserver")
            tail -f "$AIRFLOW_HOME/logs/webserver.log"
            ;;
        "scheduler")
            tail -f "$AIRFLOW_HOME/logs/scheduler.log"
            ;;
        *)
            echo "Available logs: webserver, scheduler"
            ;;
    esac
}

show_help() {
    echo "Airflow Control Script for Crypto Forecasting"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start all Airflow services"
    echo "  stop      - Stop all Airflow services"
    echo "  restart   - Restart all Airflow services"
    echo "  status    - Show status of all services"
    echo "  setup     - Setup environment only"
    echo "  logs      - Show logs (webserver|scheduler)"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start all services"
    echo "  $0 logs webserver     # Show webserver logs"
    echo "  $0 status             # Check service status"
}

case "${1:-start}" in
    "start")
        start_all
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "status")
        show_status
        ;;
    "setup")
        setup_environment
        ;;
    "logs")
        show_logs "$2"
        ;;
    "help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
