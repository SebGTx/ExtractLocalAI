import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # On cible la phase d'exécution principale du test ("call")
    if report.when == "call" and report.passed:
        # On récupère tout ce qui a été capturé (print + logging)
        captured_output = []
        for section_name, content in report.sections:
            if content.strip():
                captured_output.append(f"--- {section_name} ---\n{content}")

        if captured_output:
            full_log = "\n".join(captured_output)
            # On attache les logs aux propriétés utilisateur du rapport
            report.user_properties.append(("captured_log", full_log))
            # On force l'injection dans le rapport détaillé
            report.longrepr = full_log
