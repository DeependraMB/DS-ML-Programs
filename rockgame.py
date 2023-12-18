from random import randrange
from turtle import *
from freegames import square, vector

food = vector(0, 0)
snake = [vector(10, 0)]
aim = vector(0, -10)

def change(x, y):
    aim.x = x
    aim.y = y

def inside(head):
    return -200 < head.x < 190 and -200 < head.y < 190

def move():
    head = snake[-1].copy()
    head.move(aim)

    if not inside(head) or head in snake:
        square(head.x, head.y, 9, 'red')
        update()
        return

    snake.append(head)

    if head == food:
        print('Snake:', len(snake))
        food.x = randrange(-15, 15) * 10
        food.y = randrange(-15, 15) * 10
    else:
        snake.pop(0)

    clear()

    # Draw snake body
    for i, body in enumerate(snake):
        if i == len(snake) - 1:
            square(body.x, body.y, 9, 'green')
            draw_eyes(body.x, body.y)
            draw_tongue(body.x, body.y)
        else:
            square(body.x, body.y, 9, 'black')

    # Draw food
    square(food.x, food.y, 9, 'red')

    update()
    ontimer(move, 100)

def draw_eyes(x, y):
    eye_size = 2
    eye_distance = 4

    left_eye = vector(x - eye_distance, y + eye_distance)
    right_eye = vector(x + eye_distance, y + eye_distance)

    square(left_eye.x, left_eye.y, eye_size, 'white')
    square(right_eye.x, right_eye.y, eye_size, 'white')

def draw_tongue(x, y):
    tongue_length = 6

    tongue_start = vector(x, y - 5)
    tongue_end = vector(x, y - tongue_length)

    up()
    goto(tongue_start.x, tongue_start.y)
    down()
    goto(tongue_end.x, tongue_end.y)


setup(420, 420, 370, 0)
hideturtle()
tracer(False)
listen()
onkey(lambda: change(10, 0), 'Right')
onkey(lambda: change(-10, 0), 'Left')
onkey(lambda: change(0, 10), 'Up')
onkey(lambda: change(0, -10), 'Down')
move()
done()
