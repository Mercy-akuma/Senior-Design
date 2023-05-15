import pygame

pygame.init()

# 设置屏幕大小和标题
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Key Press Test")

# 创建Clock对象，用于控制游戏帧率
clock = pygame.time.Clock()

while True:
    # 处理游戏事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        # 监听键盘按键事件
        if event.type == pygame.KEYDOWN:
            # 输出按下的键位
            print("键位按下：", event.key)

        if event.type == pygame.KEYUP:
            # 输出松开的键位
            print("键位松开：", event.key)

    # 清屏
    screen.fill((255, 255, 255))

    # 更新屏幕
    pygame.display.update()

    # 控制游戏帧率为60FPS
    clock.tick(60)
