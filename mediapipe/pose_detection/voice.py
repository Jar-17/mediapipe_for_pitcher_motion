import pygame 

pygame.init()
pygame.mixer.init()
#pygame.mixer.music.set_volume(1.0)

#print('Path to module:',pygame._file_)
while True:
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load('kick2.mp3')
        pygame.mixer.music.play()