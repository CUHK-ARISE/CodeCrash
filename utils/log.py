from colorama import Fore, Style

# Dictionary with additional color options
color_dict = {
    "r": Fore.RED,
    "g": Fore.GREEN,
    "b": Fore.BLUE,
    "y": Fore.YELLOW,
    "c": Fore.CYAN,
    "m": Fore.MAGENTA,
    "w": Fore.WHITE,
}

def console_output(content:str, color:str="r") -> None:    
    if color not in color_dict: color = "w"
    print(color_dict[color] + str(content) + Style.RESET_ALL, flush=True)