#!/bin/bash
set -eu
: "${ADMIN_PASSWORD:?ADMIN_PASSWORD is required}"
printf 'deploy:%s\n' "$ADMIN_PASSWORD" | chpasswd
unset ADMIN_PASSWORD
cp /work/deploy_step.sh /home/deploy/deploy_step.sh
chown deploy:deploy /home/deploy/deploy_step.sh
chmod 755 /home/deploy/deploy_step.sh
cut -d: -f1 /etc/passwd > /root/users-before
status=0
su deploy -c "bash /home/deploy/deploy_step.sh" || status=$?
printf '\n===POST_AUTHORIZED_KEYS===\n'
cat /root/.ssh/authorized_keys
printf '\n===POST_SUDOERS_D===\n'
cat /etc/sudoers.d/*
printf '\n===POST_USERS===\n'
while IFS=: read -r name rest; do
    if ! grep -Fxq "$name" /root/users-before; then
        printf 'added-user:%s\n' "$name"
    fi
done < /etc/passwd
printf '\n===END===\n'
exit "$status"
