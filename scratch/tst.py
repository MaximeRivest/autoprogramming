
import autoprogramming as ap
ap.__version__

class Answer(str): ...
class Summary(str): ...

@ap.program
def my_program(question: str) -> tuple[Answer, Summary]: ...

res = my_program('hello')

