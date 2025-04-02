#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <alloc.h>
#include <graphics.h>

#define CPQ 10

#define PIXELS_MOV 20
#define JOGO_VEL 200
#define CPNUMIMPARCOR 10
#define CPNUMPARCOR 2
#define CABECACOR 4
#define FRUTA_COR 6
#define FUNDO_COR 0


// Paredes
#define PAR_VER_W 10
#define PAR_HOR_W 620
#define PAR_VER_H 420
#define PAR_HOR_H 10

// Esquerda
#define PAR_ESQ_X1 10
#define PAR_ESQ_X2 (PAR_ESQ_X1 + PAR_VER_W)
#define PAR_ESQ_Y1 50
#define PAR_ESQ_Y2 (PAR_ESQ_Y1 + PAR_VER_H)

// Direita
#define PAR_DIR_X1 PAR_ESQ_X1 + PAR_HOR_W - PAR_VER_W
#define PAR_DIR_Y1 PAR_ESQ_Y1
#define PAR_DIR_X2 (PAR_DIR_X1 + PAR_VER_W)
#define PAR_DIR_Y2 (PAR_DIR_Y1 + PAR_VER_H)

// Superior
#define PAR_SUP_X1 PAR_ESQ_X1
#define PAR_SUP_Y1 PAR_ESQ_Y1
#define PAR_SUP_X2 (PAR_SUP_X1 + PAR_HOR_W)
#define PAR_SUP_Y2 (PAR_SUP_Y1 + PAR_HOR_H)

// Inferior
#define PAR_INF_X1 PAR_ESQ_X1
#define PAR_INF_Y1 PAR_SUP_Y1 + PAR_VER_H - PAR_HOR_H
#define PAR_INF_X2 (PAR_INF_X1 + PAR_HOR_W)
#define PAR_INF_Y2 (PAR_INF_Y1 + PAR_HOR_H)


// Cobra
#define CP_INI_POS_X PAR_ESQ_X2 + (CPQ * PIXELS_MOV)
#define CP_INI_POS_Y PAR_SUP_Y2

/*=================
ESTRUTURAS
=================*/
typedef struct
{
	int corpoParteNum;					// Numero
	int corpoParteX;					// Posicao da coordenada x
	int corpoParteY;                    // Posicao da coordenada y
	int corpoParteMovX;					// Sentido da movimentacao da parte corpo no eixo x
	int corpoParteMovY;					// Sentido da movimentacao da parte corpo no eixo Y
	int corpoParteTamX;					// Tamanho da parte corpo no eixo x
	int corpoParteTamY;                 // Tamanho da parte corpo no eixo y
	enum COLORS corpoParteCor;			// Cor da parte corpo
	struct corpoParte *corpoParteLink;  // Link para a proxima parte corpo
} corpoParte;

typedef struct
{
	int frutaNum;
	int frutaX;
	int frutaY;
	int frutaTamX;
	int frutaTamY;
	enum COLORS frutaCor;
} fruta;

void graf_ini(void);
void serpente_desenhar();
void worm_mov_xy(int tecla);
void corpoParte_criar();
void fruta_criar();
void fruta_desenhar();
int paredes_corpo_limites_testar();
int serpente_comer_fruta();
void grid_desenhar();
void placar_desenhar();
void incrementarPontos();
void desenhar_game_over();
void desenhar_abertura();
void desenhar_texto(char *texto, int tam, int tamSombra, int x, int y);

corpoParte *serpenteCabeca, *serpenteCauda; // A serpente
fruta maca;									// A fruta
int pontos = 0;
int velocidade = 1;


void main(void)
{
	int i, tecla = 0;

	serpenteCabeca = NULL;
	serpenteCauda = NULL;

	maca.frutaNum = 0;

	randomize();

	graf_ini();

	desenhar_abertura();

	delay(4000);

	setfillstyle(1, FUNDO_COR);
	bar(PAR_ESQ_X2, PAR_SUP_Y2, PAR_DIR_X1, PAR_INF_Y1);
	setfillstyle(1, 15);
	bar(PAR_SUP_X1 + 310, 0, PAR_SUP_X2, PAR_SUP_Y1 - 1);
	setfillstyle(1, 2);
	bar(PAR_SUP_X1 + 312, 2, PAR_SUP_X2 - 2, PAR_SUP_Y1 - 3);
	desenhar_texto("Nibble Game", 3, 1, PAR_SUP_X1 + 347, 10);
	desenhar_texto("by Diego Giacomelli", 1, 0, PAR_SUP_X1 + 385, 38);

	setfillstyle(11, 7);
	bar(PAR_DIR_X1 + 1, PAR_DIR_Y1 + 5, PAR_DIR_X2 + 5, PAR_DIR_Y2 + 5); // Direita
	bar(PAR_INF_X1 + 5, PAR_INF_Y1 + 1, PAR_INF_X2, PAR_INF_Y2 + 5); // Inferior
	setfillstyle(1, FUNDO_COR);


	for(i = 0; i < CPQ; i++)
		corpoParte_criar();

	serpente_desenhar();
	fruta_criar();
	fruta_desenhar();
	placar_desenhar();


	for(;;)						// Loop principal do jogo
	{
		if(kbhit())				// Aquarda o pressionamento de uma tecla
		{
			fflush(stdin);		// Limpa o buffer padrao de entrada de dados
			tecla = 0;
			tecla = bioskey(0);	// Recebe a tecla pressionada
		}
		if (tecla == 283) break;// Se <ESC> for pressionada sai do jogo

		setcolor(FUNDO_COR);
		setfillstyle(1, FUNDO_COR);
		bar(serpenteCauda->corpoParteX, serpenteCauda->corpoParteY, serpenteCauda->corpoParteX + serpenteCauda->corpoParteTamX, serpenteCauda->corpoParteY + serpenteCauda->corpoParteTamY);

		worm_mov_xy(tecla);	// Avalia o valor de tecla e movimenta o flip

		if(paredes_corpo_limites_testar())
		{
			desenhar_game_over();
			break;
		}

		serpente_desenhar(); // Desenha o corpo da serpente

		if(serpente_comer_fruta())
		{
			corpoParte_criar();
			fruta_criar();
			incrementarPontos();
		}
		//grid_desenhar();

		fruta_desenhar();

		setfillstyle(1, 15);
		bar(PAR_ESQ_X1, PAR_ESQ_Y1, PAR_ESQ_X2 - 1, PAR_ESQ_Y2); // Esquerda
		bar(PAR_DIR_X1 + 1, PAR_DIR_Y1, PAR_DIR_X2, PAR_DIR_Y2); // Direita
		bar(PAR_SUP_X1, PAR_SUP_Y1, PAR_SUP_X2, PAR_SUP_Y2 - 1); // Superior
		bar(PAR_INF_X1, PAR_INF_Y1 + 1, PAR_INF_X2, PAR_INF_Y2); // Inferior

		delay(JOGO_VEL - (velocidade * 15));    // Define a velocidade do jogo
	}
	closegraph();
	clrscr();
}



/*=== Movimentacao do flip no eixo x ===*/
void worm_mov_xy(int tecla)
{
	int transX_1, transX_2, transY_1, transY_2;
	corpoParte *aux;

	transX_1 = serpenteCabeca->corpoParteX;
	transY_1 = serpenteCabeca->corpoParteY;

	switch(tecla)
	{
		case 19200:	if(serpenteCabeca->corpoParteMovX != PIXELS_MOV)
					{
						serpenteCabeca->corpoParteMovX = -PIXELS_MOV;  // Esquerda
						serpenteCabeca->corpoParteMovY= 0;
					}
					break;

		case 19712:	if(serpenteCabeca->corpoParteMovX != -PIXELS_MOV)
					{
						serpenteCabeca->corpoParteMovX = PIXELS_MOV; 	// Direita
						serpenteCabeca->corpoParteMovY= 0;
					}
					break;

		case 18432:	if(serpenteCabeca->corpoParteMovY != PIXELS_MOV)
					{
						serpenteCabeca->corpoParteMovY = -PIXELS_MOV; 	// Cima
						serpenteCabeca->corpoParteMovX= 0;
					}
					break;

		case 20480:	if(serpenteCabeca->corpoParteMovY != -PIXELS_MOV)
					{
						serpenteCabeca->corpoParteMovY = PIXELS_MOV; 	// Baixo
						serpenteCabeca->corpoParteMovX= 0;
					}
					break;
	}

	serpenteCabeca->corpoParteX +=  serpenteCabeca->corpoParteMovX;
	serpenteCabeca->corpoParteY +=  serpenteCabeca->corpoParteMovY;

	for(aux = serpenteCabeca->corpoParteLink; aux != NULL; aux = aux->corpoParteLink)
	{
		transX_2 = aux->corpoParteX;
		transY_2 = aux->corpoParteY;

		aux->corpoParteX = transX_1;
		aux->corpoParteY = transY_1;

		transX_1 = transX_2;
		transY_1 = transY_2;
	}
}


/*=== Inicializa o modo grafico ===*/
void graf_ini(void)
{
	int gdriver = DETECT, gmode;
	int coderro;
	char buf[70];

	/* inicializa o modo grafico e as variaveis relacionadas */
	initgraph(&gdriver, &gmode, "");

	coderro = graphresult();

	if (coderro != grOk)
	{
		printf("Erro no modo grafico: %s", grapherrormsg(coderro));
		getch();
		exit(1);
	}
}


void serpente_desenhar()
{
	char buf[5];
	corpoParte *aux;

	for(aux = serpenteCabeca; aux; aux = aux->corpoParteLink)
	{
		setfillstyle(9, aux->corpoParteCor);
		fillellipse(aux->corpoParteX + PIXELS_MOV/2, aux->corpoParteY + PIXELS_MOV/2, PIXELS_MOV/2, PIXELS_MOV/2);
	}
}


void corpoParte_criar()
{
	corpoParte *aux;

	aux = (corpoParte *) calloc(1, sizeof(corpoParte));
	if(aux)
	{
		aux->corpoParteNum = serpenteCauda->corpoParteNum + 1;
		aux->corpoParteX = serpenteCauda->corpoParteX - PIXELS_MOV;
		aux->corpoParteY = serpenteCauda->corpoParteY;
		aux->corpoParteTamX = PIXELS_MOV;
		aux->corpoParteTamY = PIXELS_MOV;
		aux->corpoParteMovX = PIXELS_MOV;
		aux->corpoParteMovY = 0;

		/* Se o numero da parte do corpo for impar corpoParteCor =
		CPNUMIMPARCOR senao se par corpoParteCor = CPNUMPARCOR */
		aux->corpoParteCor = aux->corpoParteNum%2 ? CPNUMIMPARCOR : CPNUMPARCOR;
		aux->corpoParteLink = NULL;
		if(serpenteCabeca == NULL)
		{
			aux->corpoParteNum = 1;
			aux->corpoParteX = CP_INI_POS_X;
			aux->corpoParteY = CP_INI_POS_Y;
			aux->corpoParteCor = CABECACOR;
			serpenteCabeca = aux;
			serpenteCauda = aux;
		}
		else
		{
			serpenteCauda->corpoParteLink = aux;
			serpenteCauda = aux;
		}

	}
}


/*=== Cria as frutas ===*/
void fruta_criar()
{
	corpoParte *aux;

	maca.frutaNum += 1;

	for(;;)
	{
		do
		{
			maca.frutaX = random(PAR_DIR_X1 - PAR_ESQ_X2);
			if(maca.frutaX % PIXELS_MOV)
				maca.frutaX += PIXELS_MOV - (maca.frutaX % PIXELS_MOV);

		} while(maca.frutaX < PAR_ESQ_X2);

		do
		{
			maca.frutaY = random(PAR_INF_Y1 - PAR_SUP_Y2);
			if(maca.frutaY % PIXELS_MOV)
				maca.frutaY += PIXELS_MOV - (maca.frutaY % PIXELS_MOV);

		} while(maca.frutaY < PAR_SUP_Y2);

		/* Se frutaX igual a alguma parte do corpo da serpente, randomiza
		novamente a frutaX */
		for(aux = serpenteCabeca; aux; aux = aux->corpoParteLink)
			if(maca.frutaX == aux->corpoParteX && maca.frutaY == aux->corpoParteY)
			{
				maca.frutaX += PIXELS_MOV;
				maca.frutaY += PIXELS_MOV;
				break;
			}

		if(maca.frutaX > PAR_ESQ_X2 && maca.frutaX < PAR_DIR_X1 && maca.frutaY > PAR_SUP_Y2 && maca.frutaY < PAR_INF_Y1) break;
	}


	maca.frutaTamX = PIXELS_MOV;
	maca.frutaTamY = PIXELS_MOV;
	maca.frutaCor =  FRUTA_COR;
}


void fruta_desenhar()
{
	setfillstyle(7, maca.frutaCor);
	fillellipse(maca.frutaX + maca.frutaTamX/2, maca.frutaY+ maca.frutaTamY/2, maca.frutaTamX/2, PIXELS_MOV/2);
}


int paredes_corpo_limites_testar()
{
	corpoParte *aux;

	if(serpenteCabeca->corpoParteX < PAR_ESQ_X2 || serpenteCabeca->corpoParteX + PIXELS_MOV > PAR_DIR_X1 || serpenteCabeca->corpoParteY < PAR_SUP_Y2 || serpenteCabeca->corpoParteY + PIXELS_MOV > PAR_INF_Y1)
		return 1;

	for(aux = serpenteCabeca->corpoParteLink; aux; aux = aux->corpoParteLink)
		if(aux->corpoParteX == serpenteCabeca->corpoParteX && aux->corpoParteY == serpenteCabeca->corpoParteY)
			return 1;

	return 0;
}


int serpente_comer_fruta()
{
	if(serpenteCabeca->corpoParteX == maca.frutaX && serpenteCabeca->corpoParteY == maca.frutaY)
		return 1;

	return 0;
}


void grid_desenhar()
{
	int i, j;

	for(i = 0; i < (PAR_DIR_X1 - PAR_ESQ_X2) / PIXELS_MOV; i++)
	{
		for(j = 0; j < (PAR_INF_Y1 - PAR_SUP_Y2) / PIXELS_MOV; j++)
		{
			rectangle(PAR_ESQ_X2 + i * PIXELS_MOV, PAR_SUP_Y2 + j * PIXELS_MOV, PAR_ESQ_X2 + PIXELS_MOV + i * PIXELS_MOV, PAR_SUP_Y2 + PIXELS_MOV + j * PIXELS_MOV);
		}
	}

}

void placar_desenhar()
{
	char buf[100];

	setfillstyle(1, 15);
	bar(PAR_SUP_X1, 0, 210, PAR_SUP_Y1 - 1);
	setfillstyle(1, 2);
	bar(PAR_SUP_X1 + 2, 0 + 2, 208, PAR_SUP_Y1 - 3);

	settextstyle(0, 0, 2);
	setfillstyle(1, 15);
	bar(PAR_SUP_X1, 0, 210, PAR_SUP_Y1 - 1);
	setfillstyle(1, 2);
	bar(PAR_SUP_X1 + 2, 0 + 2, 208, PAR_SUP_Y1 - 3);

	sprintf(buf, "Score: %i", pontos);
	desenhar_texto(buf, 2, 1, PAR_SUP_X1 + 10, 10);

	sprintf(buf, "Level: %i", velocidade);
	desenhar_texto(buf, 2, 1, PAR_SUP_X1 + 10, 30);
}

void incrementarPontos()
{
	pontos += 10;
	if((pontos % 100) == 0)
	{
		velocidade++;
	}
	placar_desenhar();
}

void desenhar_game_over()
{
	desenhar_texto("GAME OVER", 8, 5, 30, 200);
	delay(2000);
	getch();
}

void desenhar_abertura()
{
	desenhar_texto("Nibble Game", 7, 5, 15, 100);
	desenhar_texto("by Diego Giacomelli", 2, 1, 170, 250);
	desenhar_texto("www.diegogiacomelli.com.br", 3, 1, 10, 310);
}

void desenhar_texto(char *texto, int tam, int tamSombra, int x, int y)
{
	settextstyle(0, 0, tam);
	setcolor(4);
	outtextxy(x + tamSombra, y + tamSombra, texto);
	setcolor(6);
	outtextxy(x, y, texto);
}