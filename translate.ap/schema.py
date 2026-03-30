import autoprogramming as ap


class French(str): ...


@ap.program
def translate(english: str) -> French: ...
