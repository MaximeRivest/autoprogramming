# AutoProgramming

Define your inputs and outputs, and let AutoProgramming generate the code for you.

```py
import autoprogramming as ap
ap.__version__
```
```output
'0.1.0'

✓ 12ms | 1 var
```

## Example

```py
import autoprogramming as ap

class Answer(str): ...
class Summary(str): ...

@ap.program
def my_program(question: str) -> tuple[Answer, Summary]: ...

my_program.initialize(
    instructions="Given a question, produce a response.",
    descriptions=dict(
        question="The user's question",
        Answer="A concise answer",
        Summary="A brief summary",
    ),
    demos=[
        dict(question="What is the capital of France?", Answer="Paris", Summary="Paris is the capital"),
    ],
    model="gpt-4.1",
    adapter=XMLAdapter(),
    temperature=0.7,
    max_tokens=100,
)

my_program.save("my_program.ap")
```

In another session we can load the program and use it:

```py
import autoprogramming as ap
my_program = ap.load("my_program.ap")
res = my_program("What is the capital of France?")
```

We could keep a lot more things unspecified, and let AutoProgramming figure them out. For example, we could just specify the types of the inputs and outputs, and let AutoProgramming figure out how to use them:

```py
import autoprogramming as ap

class French(str): ...

@ap.program
def translate_to_french(english: str) -> French: ...

translate.optimize(data = train_df)

translate.save("translate.ap")

translate('Hello, how are you?')
```
```output
'Bonjour, comment ça va?'
```




```py
type Response = str

@ap.program
def my_program(question: str) -> Response:
    pass

res = my_program('hello')
```

```py
@ap.program
def my_program(question: str) -> Annotated[str, "response"]:
    pass

res = my_program('hello')
```

```py
class Response(str): ...

@ap.program
def my_program(question: str) -> Response: ...

res = my_program('hello')
```


```py
class Answer(str): ...
class Summary(str): ...

@ap.program
def my_program(question: str) -> (Answer, Summary): ...

res = my_program('hello')
```



```py
class Answer(str):
    """A concise answer"""

class Summary(str):
    """A brief summary"""

@ap.program
def my_program(question: str) -> tuple[Answer, Summary]:
    """question: The user's question"""

```