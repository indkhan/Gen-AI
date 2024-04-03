import pygame

# Initialize the game engine
pygame.init()

# Set up the game screen
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define the character
character_x = 100
character_y = HEIGHT - 50

# Define the obstacle
obstacle_x = WIDTH - 50
obstacle_y = HEIGHT - 20

# Clock speed
clock_speed = 5

# Game loop flag
running = True

# Main game loop
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update character position
    character_x += clock_speed * 1
    character_y -= clock_speed * 1

    # Check if character has collided with the obstacle
    if character_x >= obstacle_x and character_y <= obstacle_y:
        running = False

    # Draw the screen
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 0, 0), (character_x, character_y, 50, 50))
    pygame.draw.rect(screen, (255, 255, 0), (obstacle_x, obstacle_y, 50, 50))

    # Update the display
    pygame.display.update()

    # Clock tick
    pygame.time.Clock().tick(clock_speed)

# Quit pygame
pygame.quit()
