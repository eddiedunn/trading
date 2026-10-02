# Deploys run from the Mac: the playbooks read secrets from gopass.
ANSIBLE = cd deploy/ansible && ansible-playbook

.PHONY: test deploy deploy-backtest deploy-postgres deploy-paper deploy-live retire-legacy smoketest

test:
	uv run --extra dev pytest -q

deploy:
	$(ANSIBLE) playbooks/deploy_all.yml

deploy-backtest:
	$(ANSIBLE) playbooks/deploy_backtest_api.yml

deploy-postgres:
	$(ANSIBLE) playbooks/deploy_trading_postgres.yml

deploy-paper:
	$(ANSIBLE) playbooks/deploy_trading_paper_arena.yml

deploy-live:
	$(ANSIBLE) playbooks/deploy_trading_live.yml

retire-legacy:
	$(ANSIBLE) playbooks/retire_legacy.yml

smoketest:
	scripts/smoketest.sh
