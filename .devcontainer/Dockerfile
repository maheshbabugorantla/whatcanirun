# syntax=docker/dockerfile:1
#
# Dev container for whatcanirun.
# Pattern: Anthropic's reference devcontainer + project-specific tooling.
#
# Run `claude --dangerously-skip-permissions` inside this container.
# Host filesystem and network are isolated; the egress firewall (init-firewall.sh)
# limits outbound traffic to allowlisted domains.

FROM node:24-bookworm

ARG TZ=Etc/UTC
ENV TZ="$TZ"
ENV DEBIAN_FRONTEND=noninteractive

# ---- System tooling ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      wget \
      gnupg \
      git \
      git-lfs \
      openssh-client \
      sudo \
      tzdata \
      unzip \
      zip \
      zsh \
      less \
      vim \
      nano \
      jq \
      tree \
      htop \
      ripgrep \
      fd-find \
      iptables \
      ipset \
      dnsutils \
      iputils-ping \
      net-tools \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev \
   && rm -rf /var/lib/apt/lists/*

# ---- uv (modern Python package manager) -------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# ---- GitHub CLI (gh) --------------------------------------------------------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# ---- Non-root user setup ----------------------------------------------------
# node user comes from base image with UID 1000.
ARG USERNAME=node
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN mkdir -p /commandhistory \
 && touch /commandhistory/.zsh_history \
 && chown -R $USERNAME:$USERNAME /commandhistory

RUN echo "$USERNAME ALL=(root) NOPASSWD:/usr/local/bin/init-firewall.sh" \
      > /etc/sudoers.d/$USERNAME-firewall \
 && chmod 0440 /etc/sudoers.d/$USERNAME-firewall

# ---- Persistent caches/configs ----------------------------------------------
RUN mkdir -p /home/$USERNAME/.claude \
             /home/$USERNAME/.cache \
             /home/$USERNAME/.local/share \
 && chown -R $USERNAME:$USERNAME /home/$USERNAME

# ---- Claude Code CLI --------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code

# ---- Bun (required by claude-mem plugin hooks) ------------------------------
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash

# ---- iptables: use legacy backend inside containers -------------------------
RUN update-alternatives --set iptables  /usr/sbin/iptables-legacy  && \
    update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

# ---- Firewall init script ---------------------------------------------------
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod 0755 /usr/local/bin/init-firewall.sh

# ---- Python 3.12 via uv -----------------------------------------------------
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.12 \
 && chmod -R a+rX /opt/uv-python

# ---- Default user + shell ---------------------------------------------------
USER $USERNAME
WORKDIR /workspace

ENV SHELL=/bin/zsh
ENV CLAUDE_CONFIG_DIR=/home/$USERNAME/.claude
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv-linux
ENV VIRTUAL_ENV=/workspace/.venv-linux
ENV PATH="/workspace/.venv-linux/bin:$PATH"

RUN echo 'export HISTFILE=/commandhistory/.zsh_history' >> /home/$USERNAME/.zshrc \
 && echo 'export HISTSIZE=10000' >> /home/$USERNAME/.zshrc \
 && echo 'export SAVEHIST=10000' >> /home/$USERNAME/.zshrc \
 && echo 'setopt INC_APPEND_HISTORY SHARE_HISTORY' >> /home/$USERNAME/.zshrc \
 && echo 'export UV_PROJECT_ENVIRONMENT=/workspace/.venv-linux' >> /home/$USERNAME/.zshrc \
 && echo 'export VIRTUAL_ENV=/workspace/.venv-linux' >> /home/$USERNAME/.zshrc \
 && echo 'export PATH="/workspace/.venv-linux/bin:$PATH"' >> /home/$USERNAME/.zshrc

# Rewrite SSH git URLs to HTTPS so plugin marketplace clones work without SSH keys
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

CMD ["sleep", "infinity"]
