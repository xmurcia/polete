module.exports = {
  apps: [{
    name: 'elonbot-real',
    script: 'main.py',
    args: '--real-trading',
    interpreter: 'python3',
    cwd: '/Users/xavi.murcia/Desktop/poly-gemini',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '500M',
    env: {
      PYTHONUNBUFFERED: '1'
    },
    error_file: 'logs/pm2-real-error.log',
    out_file: 'logs/pm2-real-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    merge_logs: true,
    min_uptime: '10s',
    max_restarts: 10,
    restart_delay: 5000
  }]
};
