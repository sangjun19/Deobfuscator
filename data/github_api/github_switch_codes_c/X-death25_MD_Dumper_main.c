/*

MD Dumper CLI Version
X-death - 01/2025

MD Dumper SDL GUI Version
Jackobo Le Chocobo (Akina Usagi) - 31/08/2024

*/

//Operating System Libraries and variables declarations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//MD Dumper Functions
#include "md_dumper_main.h"
#include "md_dumper_read.h"
#include "md_dumper_write.h"

int main(int argc, char *argv[])
{
	SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO); //Display informations on console
	
	if(argc == 1)
	{
		Display_Help(argv[0]);
		return 1;
	}
	else if (strcmp(argv[1], "-help") == 0)
	{
		Display_Help(argv[0]);
		return 1;
	}

	//Using GUI Mode ?
	int lang=0;
	if(strcmp(argv[1], "-gui") == 0)
	{
		use_gui=1;
		lang=0;
	}
	else if(strcmp(argv[1], "-gui_fr") == 0)
	{
		use_gui=1;
		lang=1;
	}
	
	if (use_gui==1)
	{  
		int gui_scale=2;
		int quit = 0;
		SDL_Event event;
		int mouse_x, mouse_y;

		//Init Window
		SDL_Init(SDL_INIT_VIDEO);
		SDL_Window * window = SDL_CreateWindow("MD Dumper version January 2025", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 424*gui_scale, 240*gui_scale, 0);
		SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
		
		SDL_Surface * image1;
		SDL_Surface * image2;
		SDL_Surface * image3;
		SDL_Surface * image4;
		SDL_Surface * image5;
		
		//Create Background Texture
		if(lang==1)
		{
			#if defined(_WIN32)
				image1 = IMG_Load(".\\images\\opts_background_read_default_FR.png");
				image2 = IMG_Load(".\\images\\opts_background_read_ram_manual_FR.png");
				image3 = IMG_Load(".\\images\\opts_background_read_rom_manual_FR.png");
				image4 = IMG_Load(".\\images\\opts_background_write_default_FR.png");
				image5 = IMG_Load(".\\images\\opts_background_write_ram_FR.png");
			#else
				image1 = IMG_Load("./images/opts_background_read_default_FR.png");
				image2 = IMG_Load("./images/opts_background_read_ram_manual_FR.png");
				image3 = IMG_Load("./images/opts_background_read_rom_manual_FR.png");
				image4 = IMG_Load("./images/opts_background_write_default_FR.png");
				image5 = IMG_Load("./images/opts_background_write_ram_FR.png");
			#endif
		}
		else
		{
			#if defined(_WIN32)
				image1 = IMG_Load(".\\images\\opts_background_read_default.png");
				image2 = IMG_Load(".\\images\\opts_background_read_ram_manual.png");
				image3 = IMG_Load(".\\images\\opts_background_read_rom_manual.png");
				image4 = IMG_Load(".\\images\\opts_background_write_default.png");
				image5 = IMG_Load(".\\images\\opts_background_write_ram.png");
			#else
				image1 = IMG_Load("./images/opts_background_read_default.png");
				image2 = IMG_Load("./images/opts_background_read_ram_manual.png");
				image3 = IMG_Load("./images/opts_background_read_rom_manual.png");
				image4 = IMG_Load("./images/opts_background_write_default.png");
				image5 = IMG_Load("./images/opts_background_write_ram.png");
			#endif			
		}

		while (quit==0)
		{
			SDL_Texture * texture;
			if(gui_tab_mode==0 && (dump_mode==0 || dump_mode==2))		texture = SDL_CreateTextureFromSurface(renderer, image1);
			else if(gui_tab_mode==0 && data_type==1 && dump_mode==1)	texture = SDL_CreateTextureFromSurface(renderer, image2);
			else if(gui_tab_mode==0 && data_type==0 && dump_mode==1)	texture = SDL_CreateTextureFromSurface(renderer, image3);
			else if(gui_tab_mode==1 && data_type==0)			texture = SDL_CreateTextureFromSurface(renderer, image4);
			else if(gui_tab_mode==1 && data_type==1)			texture = SDL_CreateTextureFromSurface(renderer, image5);
			
			SDL_RenderCopy(renderer, texture, NULL, NULL);
			SDL_SetRenderDrawColor(renderer, 250, 173, 5, 255);

			if(gui_tab_mode==0)
			{
				switch(data_type)
				{
					case 0: //ROM								
						for (int x = 224*gui_scale; x <=230*gui_scale; x++)
							for (int y = 67*gui_scale; y <=73*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
					case 1: //RAM
						for (int x = 270*gui_scale; x <=276*gui_scale; x++)
							for (int y = 67*gui_scale; y <=73*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
				}
				
				switch(dump_mode)
				{
					case 0: //Auto
						for (int x = 224*gui_scale; x <=230*gui_scale; x++)
							for (int y = 84*gui_scale; y <=90*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
					case 1: //Manual
						for (int x = 270*gui_scale; x <=276*gui_scale; x++)
							for (int y = 84*gui_scale; y <=90*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
								
						switch(data_type)
						{
							case 0: //ROM								
								switch(dump_rom_size_opts)
								{
									case 0: //32
										for (int x = 20*gui_scale; x <=26*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 1: //64
										for (int x = 66*gui_scale; x <=72*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 2: //128
										for (int x = 112*gui_scale; x <=118*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 3: //256
										for (int x = 158*gui_scale; x <=164*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 4: //512
										for (int x = 204*gui_scale; x <=210*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 5: //1024
										for (int x = 20*gui_scale; x <=26*gui_scale; x++)
											for (int y = 157*gui_scale; y <=163*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 6: //2048
										for (int x = 66*gui_scale; x <=72*gui_scale; x++)
											for (int y = 157*gui_scale; y <=163*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 7: //4096
										for (int x = 112*gui_scale; x <=118*gui_scale; x++)
											for (int y = 157*gui_scale; y <=163*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 8: //8192
										for (int x = 158*gui_scale; x <=164*gui_scale; x++)
											for (int y = 157*gui_scale; y <=163*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
								}
								
								switch(dump_cart_mode_opts)
								{
									case 0: //GG
										for (int x = 250*gui_scale; x <=256*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 1: //MD
										for (int x = 296*gui_scale; x <=302*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 2: //SMS
										for (int x = 342*gui_scale; x <=348*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
								}
								break;
							case 1: //RAM
								for (int x = 270*gui_scale; x <=276*gui_scale; x++)
									for (int y = 67*gui_scale; y <=73*gui_scale; y++)
										SDL_RenderDrawPoint(renderer, x, y);
								
								switch(dump_sram_size_opts)
								{
									case 0: //8192
										for (int x = 20*gui_scale; x <=26*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 1: //32768
										for (int x = 66*gui_scale; x <=72*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
								}
								
								switch(dump_sram_type_opts)
								{
									case 0: //Parallel SRAM
										for (int x = 250*gui_scale; x <=256*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 1: //Serial I2C
										for (int x = 342*gui_scale; x <=348*gui_scale; x++)
											for (int y = 140*gui_scale; y <=146*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
									case 2: //Serial SPI
										for (int x = 250*gui_scale; x <=256*gui_scale; x++)
											for (int y = 157*gui_scale; y <=163*gui_scale; y++)
												SDL_RenderDrawPoint(renderer, x, y);
										break;
								}
								
								break;
						}		
						break;		
					case 2: //Bankswitch
						for (int x = 329*gui_scale; x <=335*gui_scale; x++)
							for (int y = 84*gui_scale; y <=90*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
				}		
			}
			else if(gui_tab_mode==1)
			{
				if(data_type==1)
				{
					switch(dump_sram_type_opts)
					{
						case 0: //Parallel SRAM
							for (int x = 250*gui_scale; x <=256*gui_scale; x++)
								for (int y = 140*gui_scale; y <=146*gui_scale; y++)
									SDL_RenderDrawPoint(renderer, x, y);
							break;
						case 1: //Serial I2C
							for (int x = 342*gui_scale; x <=348*gui_scale; x++)
								for (int y = 140*gui_scale; y <=146*gui_scale; y++)
									SDL_RenderDrawPoint(renderer, x, y);
							break;
						case 2: //Serial SPI
							for (int x = 250*gui_scale; x <=256*gui_scale; x++)
								for (int y = 157*gui_scale; y <=163*gui_scale; y++)
									SDL_RenderDrawPoint(renderer, x, y);
							break;
					}
				}
				
				switch(erase_or_write)
				{
					case 0: //Erase								
						for (int x = 224*gui_scale; x <=230*gui_scale; x++)
							for (int y = 67*gui_scale; y <=73*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
					case 1: //Write
						for (int x = 282*gui_scale; x <=288*gui_scale; x++)
							for (int y = 67*gui_scale; y <=73*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
				}	
				
				switch(data_type)
				{
					case 0: //ROM								
						for (int x = 224*gui_scale; x <=230*gui_scale; x++)
							for (int y = 84*gui_scale; y <=90*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
					case 1: //RAM
						for (int x = 282*gui_scale; x <=288*gui_scale; x++)
							for (int y = 84*gui_scale; y <=90*gui_scale; y++)
								SDL_RenderDrawPoint(renderer, x, y);
						break;
				}	
			}
					
			//Display Texture	
			SDL_RenderPresent(renderer);
		
			SDL_GetMouseState(&mouse_x, &mouse_y);
			SDL_WaitEvent(&event);
				 
			switch (event.type)		//Window Events according to mouse positions and left click on this Window
			{
				case SDL_QUIT:
					quit = 1;
					SDL_DestroyTexture(texture);
					SDL_FreeSurface(image1);
					SDL_FreeSurface(image2);
					SDL_FreeSurface(image3);
					SDL_FreeSurface(image4);
					SDL_FreeSurface(image5);
					SDL_DestroyRenderer(renderer);
					SDL_DestroyWindow(window);
					SDL_Quit();
					return 1;
				case SDL_MOUSEBUTTONDOWN:
					if(mouse_x>=18*gui_scale  && mouse_x<=108*gui_scale) 
					{
						if (mouse_y>=46*gui_scale  && mouse_y<=58*gui_scale)
							gui_tab_mode = 0;
					}
					else if(mouse_x>=117*gui_scale  && mouse_x<=207*gui_scale) 
					{
						if (mouse_y>=46*gui_scale && mouse_y<=58*gui_scale)
							gui_tab_mode = 1;
					}
					else if(mouse_x>=337*gui_scale && mouse_x<=419*gui_scale)
					{
						if(mouse_y>=214*gui_scale && mouse_y<=235*gui_scale)	//Launch
						{
							quit = 1;
							SDL_DestroyTexture(texture);
							SDL_FreeSurface(image1);
							SDL_FreeSurface(image2);
							SDL_FreeSurface(image3);
							SDL_FreeSurface(image4);
							SDL_FreeSurface(image5);
							SDL_DestroyRenderer(renderer);
							SDL_DestroyWindow(window);
							SDL_Quit();
							break;
						}
					}
					
					if(gui_tab_mode==0 && (dump_mode==0 || dump_mode==2))
					{
						if(mouse_x>=222*gui_scale  && mouse_x<=232*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 0;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 0;
						}
						else if(mouse_x>=268*gui_scale && mouse_x<=278*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 1;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 1;
						}
						else if(mouse_x>=327*gui_scale && mouse_x<=337*gui_scale) 
						{
							if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 2;
						}
					}
					else if(gui_tab_mode==0 && data_type==1 && dump_mode==1)
					{
						if(mouse_x>=222*gui_scale && mouse_x<=232*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 0;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 0;
						}
						else if(mouse_x>=268*gui_scale && mouse_x<=278*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 1;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 1;
						}
						else if(mouse_x>=327*gui_scale && mouse_x<=337*gui_scale) 
						{
							if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 2;
						}
						else if(mouse_x>=18*gui_scale && mouse_x<=28*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_sram_size_opts = 0;
						}
						else if(mouse_x>=64*gui_scale && mouse_x<=74*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_sram_size_opts = 1;
						}
						else if(mouse_x>=248*gui_scale && mouse_x<=258*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_sram_type_opts = 0;
							else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
								dump_sram_type_opts = 2;
						}
						else if(mouse_x>=340*gui_scale && mouse_x<=350*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_sram_type_opts = 1;
						}						
					}
					else if(gui_tab_mode==0 && data_type==0 && dump_mode==1)
					{
						if(mouse_x>=222*gui_scale && mouse_x<=232*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 0;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 0;
						}
						else if(mouse_x>=268*gui_scale && mouse_x<=278*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								data_type = 1;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 1;
						}
						else if(mouse_x>=327*gui_scale && mouse_x<=337*gui_scale) 
						{
							if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								dump_mode = 2;
						}
						else if(mouse_x>=18*gui_scale && mouse_x<=28*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_rom_size_opts = 0;
							else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
								dump_rom_size_opts = 5;
						}
						else if(mouse_x>=64*gui_scale && mouse_x<=74*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_rom_size_opts = 1;
							else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
								dump_rom_size_opts = 6;
						}
						else if(mouse_x>=110*gui_scale && mouse_x<=120*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_rom_size_opts = 2;
							else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
								dump_rom_size_opts = 7;
						}
						else if(mouse_x>=156*gui_scale && mouse_x<=166*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_rom_size_opts = 3;
							else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
								dump_rom_size_opts = 8;
						}
						else if(mouse_x>=202*gui_scale && mouse_x<=212*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_rom_size_opts = 4;
						}
						else if(mouse_x>=248*gui_scale && mouse_x<=258*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_cart_mode_opts = 0;
						}
						else if(mouse_x>=294*gui_scale && mouse_x<=304*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_cart_mode_opts = 1;
						}
						else if(mouse_x>=340*gui_scale && mouse_x<=350*gui_scale) 
						{
							if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
								dump_cart_mode_opts = 2;
						}
					}
					else if(gui_tab_mode==1)
					{
						if(data_type==1)
						{
							if(mouse_x>=248*gui_scale && mouse_x<=258*gui_scale) 
							{
								if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
									dump_sram_type_opts = 0;
								else if (mouse_y>=155*gui_scale && mouse_y<=165*gui_scale)
									dump_sram_type_opts = 2;
							}
							else if(mouse_x>=340*gui_scale && mouse_x<=350*gui_scale) 
							{
								if (mouse_y>=138*gui_scale && mouse_y<=148*gui_scale)
									dump_sram_type_opts = 1;
							}		
						}
						if(mouse_x>=222*gui_scale && mouse_x<=232*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								erase_or_write = 0;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								data_type = 0;
						}
						else if(mouse_x>=280*gui_scale && mouse_x<=290*gui_scale) 
						{
							if (mouse_y>=65*gui_scale && mouse_y<=75*gui_scale)
								erase_or_write = 1;
							else if (mouse_y>=82*gui_scale && mouse_y<=92*gui_scale)
								data_type = 1;
						}
					}
					break;
			}
		}
	}
	else
	{
		SDL_Log("\n");
		SDL_Log("----------------------------------------------------------------\n");
		SDL_Log("8b   d8 888b.      888b. 8    8 8b   d8 888b. 8888 888b. \n");
		SDL_Log("8YbmdP8 8   8      8   8 8    8 8YbmdP8 8  .8 8www 8  .8 \n");
		SDL_Log("8     8 8   8 wwww 8   8 8b..d8 8     8 8wwP' 8    8wwK' \n");
		SDL_Log("8     8 888P'      888P' `Y88P' 8     8 8     8888 8  Yb \n");
		SDL_Log("----------------------------------------------------------------\n");
		SDL_Log("\n");
		SDL_Log("Release : 02 Nov. 2024 \n");
		SDL_Log("\n");
	}

	//LibUsb : Init & Detect
	if(Detect_Device()!=0)		return 1;
	
	//LibCsv : Init & Open files
	if(Open_CSV_Files()!=0)		return 1;

	//Read Game Header/Infos
	Game_Header_Infos();

	// Vérifier le nombre d'arguments
	if(use_gui==0)						//Vérifier que nous utilisons le mode CLI
	{
		//Lecture de la ROM
		if (strcmp(argv[1], "-read") == 0)
		{
			if (strcmp(argv[2], "auto")==0)
			{
				gui_tab_mode=0;
				data_type=0;
				dump_mode=0;
			}
			else  if (strcmp(argv[2], "manual")==0)
			{
				gui_tab_mode=0;
				data_type=0;
				dump_mode=1;
				
				if (strcmp(argv[3], "32") == 0)						dump_rom_size_opts = 0;
				else if (strcmp(argv[3], "64") == 0)					dump_rom_size_opts = 1;
				else if (strcmp(argv[3], "128") == 0)					dump_rom_size_opts = 2;
				else if (strcmp(argv[3], "256") == 0)					dump_rom_size_opts = 3;
				else if (strcmp(argv[3], "512") == 0)					dump_rom_size_opts = 4;
				else if (strcmp(argv[3], "1024") == 0)					dump_rom_size_opts = 5;
				else if (strcmp(argv[3], "2048") == 0)					dump_rom_size_opts = 6;
				else if (strcmp(argv[3], "4096") == 0)					dump_rom_size_opts = 7;
				else if (strcmp(argv[3], "8192") == 0)					dump_rom_size_opts = 8;
				else
				{
					SDL_Log("You must write one of the following values to set the game size : 32, 64, 128, 256, 512, 1024, 2048, 4096.\n");
					return 1;
				}
				
				if (strcmp(argv[4], "gg") == 0)						dump_cart_mode_opts = 0;
				else if (strcmp(argv[4], "md") == 0)					dump_cart_mode_opts = 1;
				else if (strcmp(argv[4], "sms") == 0)					dump_cart_mode_opts = 2;
				else
				{
					SDL_Log("You must write one of the following values to set the cartridge type : gg, md or sms.\n");
					return 1;
				}
			}
			else if (strcmp(argv[2], "bankswitch") == 0)
			{
				gui_tab_mode=0;
				data_type=0;
				dump_mode=2;
			}
			else
			{
				SDL_Log("You must select 'a' (Auto), 'b' (Bankswitch) or 'm' (Manual).\n");
				return 1;
			}
		}
		else if (strcmp(argv[1], "-backup") == 0)
		{
			if (strcmp(argv[2], "auto")==0)
			{
				gui_tab_mode=0;
				data_type=1;
				dump_mode=0;
			}
			else  if (strcmp(argv[2], "manual")==0)
			{
				gui_tab_mode=0;
				data_type=1;
				dump_mode=1;
				
				if (strcmp(argv[3], "8192") == 0)					dump_sram_size_opts = 0;
				else if (strcmp(argv[3], "32768") == 0)					dump_sram_size_opts = 1;
				else
				{
					SDL_Log("You must write one of the following values to set the game size : 8192, 32768.\n");
					return 1;
				}
				
				if (strcmp(argv[4], "parallel_sram") == 0)				dump_sram_type_opts = 0;
				else if (strcmp(argv[4], "serial_i2c") == 0)				dump_sram_type_opts = 1;
				else if (strcmp(argv[4], "serial_spi") == 0)				dump_sram_type_opts = 2;
				else
				{
					SDL_Log("You must write one of the following values to set the cartridge type : parallel_sram, serial_i2c or serial_spi.\n");
					return 1;
				}
			}
			else if (strcmp(argv[2], "bankswitch") == 0)
			{
				gui_tab_mode=0;
				data_type=1;
				dump_mode=2;
			}
			else
			{
				SDL_Log("You must select 'a' (Auto), 'b' (Bankswitch) or 'm' (Manual).\n");
				return 1;
			}
		}
		else if (strcmp(argv[1], "-erase_flash") == 0)
		{
			gui_tab_mode=1;
			data_type=0;
			erase_or_write=0;
		}
		else if (strcmp(argv[1], "-write_flash") == 0)
		{
			gui_tab_mode=1;
			data_type=0;
			erase_or_write=1;
		}
		else if (strcmp(argv[1], "-erase_memory") == 0)
		{
			gui_tab_mode=1;
			data_type=1;
			erase_or_write=0;
			
			if (strcmp(argv[2], "serial_spi") == 0)						dump_sram_type_opts = 2;
			else if (strcmp(argv[2], "serial_i2c") == 0)					dump_sram_type_opts = 1;
			else if (strcmp(argv[2], "parallel_sram") == 0)					dump_sram_type_opts = 0;
			else
			{
				SDL_Log("You must write one of the following values to select the save type : serial_spi, serial_i2c, parallel_sram.\n");
				return 1;
			}
		}
		else if (strcmp(argv[1], "-write_memory") == 0)
		{
			gui_tab_mode=1;
			data_type=1;
			erase_or_write=1;
			
			if (strcmp(argv[2], "serial_spi") == 0)						dump_sram_type_opts = 2;
			else if (strcmp(argv[2], "serial_i2c") == 0)					dump_sram_type_opts = 1;
			else if (strcmp(argv[2], "parallel_sram") == 0)					dump_sram_type_opts = 0;
			else
			{
				SDL_Log("You must write one of the following values to select the save type : serial_spi, serial_i2c, parallel_sram.\n");
				return 1;
			}
		}
		//Erreur
		else
		{
			SDL_Log("You must write '-read', '-backup', '-erase_flash', '-write_flash', '-erase_memory' or '-write_memory' .\n");
			return 1;
		}
	}

	if ( gui_tab_mode==0 && data_type==0 && dump_mode==0 )				Read_ROM_Auto();
	else if ( gui_tab_mode==0 && data_type==0 && dump_mode==1 )			Read_ROM_Manual();
	else if ( gui_tab_mode==0 && data_type==0 && dump_mode==2 )			Read_ROM_Bankswitch();
	else if ( gui_tab_mode==0 && data_type==1 && dump_mode==0 )			Read_RAM_Auto();
	else if ( gui_tab_mode==0 && data_type==1 && dump_mode==1 )			Read_RAM_Manual();
	else if ( gui_tab_mode==0 && data_type==1 && dump_mode==2 )			Read_RAM_Bankswitch();
	else if ( gui_tab_mode==1 && data_type==0 && erase_or_write==0 )		Erase_Flash();
	else if ( gui_tab_mode==1 && data_type==0 && erase_or_write==1 )		Write_Flash();
	else if ( gui_tab_mode==1 && data_type==1 && erase_or_write==0 )		Erase_RAM();
	else if ( gui_tab_mode==1 && data_type==1 && erase_or_write==1 )		Write_RAM();
	
	libusb_release_interface(handle,if_num);
	libusb_close(handle);
	libusb_exit(context);
	
	return 0;
}


