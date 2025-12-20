# Auto Trader Deployment Guide

This folder contains scripts to easily deploy the Auto Trader to a cloud VM (Ubuntu/Debian recommended).

## One-Click Deployment Steps

1.  **Connect to your VM** via SSH.
2.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd auto-trader-okx-lt
    ```
3.  **Run the Setup Script** (First time only):
    This installs Docker, Docker Compose, and other dependencies.
    ```bash
    chmod +x deploy/setup_vm.sh
    ./deploy/setup_vm.sh
    ```
    *Note: If you just installed Docker, you might need to logout and login again for group permissions to take effect.*

4.  **Run the Start Script**:
    This configures your environment (asking for API keys if needed) and starts the bot.
    ```bash
    chmod +x deploy/start.sh
    ./deploy/start.sh
    ```

## Managing the Bot

-   **View Logs**:
    ```bash
    docker-compose logs -f
    ```
-   **Stop the Bot**:
    ```bash
    docker-compose down
    ```
-   **Restart**:
    ```bash
    docker-compose restart
    ```
-   **Update Code**:
    ```bash
    git pull
    ./deploy/start.sh
    ```

## Configuration

-   **Credentials**: Stored in `.env`.
-   **Strategy Settings**: Stored in `config/settings.yaml`. You can edit this file and restart the bot to apply changes.
