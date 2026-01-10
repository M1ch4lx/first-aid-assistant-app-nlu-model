# Uniwersalne narzędzie do implementacji backendu systemu konwersacyjnego

## Wymagania

- Python 3.11
- Conda / UV (Opcjonalne)

## Instalacja bibliotek

Python Venv / Conda: `pip install -r requirements.txt`\
UV: `uv pip install -r requirements.txt`

## Aplikacja asystenta pierwszej pomocy

Zaimplementowane narzędzie nie ogranicza się jedynie do zaimplementowania usługi sieciowej dla aplikacji asystenta pierwszej pomocy. Możliwe jest tworzenie za jego pomocą systemów konwersacyjnych o dowolnym zastosowaniu.

W katalogu głównym projektu znajduje się katalog `bot`, w który znajdują się pliki asystenta do udzielania pierwszej pomocy zbudowanego za pomocą tego narzędzia.

W celu uruchomienia usługi webowej, z której będzie korzystać aplikacja, wystarczy w katalogu głównym wykonać polecenie: `python bot.py start_server`.

## Budowanie własnego systemu konwersacyjnego

1. Z katalogu głównego należy usunąć katalog `bot` (jeżeli istnieje).

2. Utworzenie nowego projektu: `python bot.py init`.

3. Pliki z rozszerzeniem `.yml` definiują działanie systemu.

4. Po każdym zmodyfikowaniu pliku `nlu.yml` należy wykonać trening modelu NLU poleceniem `python bot.py train`.

5. Aby przetestować samą klasyfikacje intencji, czyli sam model NLU, można użyć polecenia `python bot.py nlu`.

6. Aby przetestować lokalnie jak działa utworzony system, należy skorzystać z komendy `python bot.py run`.

7. Jeżeli system działa zgodnie z oczekiwaniami można go uruchomić na serwerze poleceniem `python bot.py start_server`.