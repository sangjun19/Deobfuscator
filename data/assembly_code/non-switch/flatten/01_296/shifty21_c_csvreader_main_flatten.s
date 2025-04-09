	.file	"shifty21_c_csvreader_main_flatten.c"
	.text
	.globl	_TIG_IZ_dk3g_argv
	.bss
	.align 8
	.type	_TIG_IZ_dk3g_argv, @object
	.size	_TIG_IZ_dk3g_argv, 8
_TIG_IZ_dk3g_argv:
	.zero	8
	.globl	books
	.align 32
	.type	books, @object
	.size	books, 8192
books:
	.zero	8192
	.globl	_TIG_IZ_dk3g_argc
	.align 4
	.type	_TIG_IZ_dk3g_argc, @object
	.size	_TIG_IZ_dk3g_argc, 4
_TIG_IZ_dk3g_argc:
	.zero	4
	.globl	_TIG_IZ_dk3g_envp
	.align 8
	.type	_TIG_IZ_dk3g_envp, @object
	.size	_TIG_IZ_dk3g_envp, 8
_TIG_IZ_dk3g_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC1:
	.string	"Please privide input and output file names with the binary."
	.align 8
.LC2:
	.string	"Reading the file %s and output will be %s\n"
.LC6:
	.string	"Run Completed!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$608, %rsp
	movl	%edi, -580(%rbp)
	movq	%rsi, -592(%rbp)
	movq	%rdx, -600(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	pxor	%xmm0, %xmm0
	movsd	%xmm0, books(%rip)
	movl	$0, 8+books(%rip)
	movl	$0, 12+books(%rip)
	movl	$0, 16+books(%rip)
	movl	$0, 20+books(%rip)
	movl	$0, 24+books(%rip)
	movl	$0, 28+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 32+books(%rip)
	movl	$0, 40+books(%rip)
	movl	$0, 44+books(%rip)
	movl	$0, 48+books(%rip)
	movl	$0, 52+books(%rip)
	movl	$0, 56+books(%rip)
	movl	$0, 60+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 64+books(%rip)
	movl	$0, 72+books(%rip)
	movl	$0, 76+books(%rip)
	movl	$0, 80+books(%rip)
	movl	$0, 84+books(%rip)
	movl	$0, 88+books(%rip)
	movl	$0, 92+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 96+books(%rip)
	movl	$0, 104+books(%rip)
	movl	$0, 108+books(%rip)
	movl	$0, 112+books(%rip)
	movl	$0, 116+books(%rip)
	movl	$0, 120+books(%rip)
	movl	$0, 124+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 128+books(%rip)
	movl	$0, 136+books(%rip)
	movl	$0, 140+books(%rip)
	movl	$0, 144+books(%rip)
	movl	$0, 148+books(%rip)
	movl	$0, 152+books(%rip)
	movl	$0, 156+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 160+books(%rip)
	movl	$0, 168+books(%rip)
	movl	$0, 172+books(%rip)
	movl	$0, 176+books(%rip)
	movl	$0, 180+books(%rip)
	movl	$0, 184+books(%rip)
	movl	$0, 188+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 192+books(%rip)
	movl	$0, 200+books(%rip)
	movl	$0, 204+books(%rip)
	movl	$0, 208+books(%rip)
	movl	$0, 212+books(%rip)
	movl	$0, 216+books(%rip)
	movl	$0, 220+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 224+books(%rip)
	movl	$0, 232+books(%rip)
	movl	$0, 236+books(%rip)
	movl	$0, 240+books(%rip)
	movl	$0, 244+books(%rip)
	movl	$0, 248+books(%rip)
	movl	$0, 252+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 256+books(%rip)
	movl	$0, 264+books(%rip)
	movl	$0, 268+books(%rip)
	movl	$0, 272+books(%rip)
	movl	$0, 276+books(%rip)
	movl	$0, 280+books(%rip)
	movl	$0, 284+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 288+books(%rip)
	movl	$0, 296+books(%rip)
	movl	$0, 300+books(%rip)
	movl	$0, 304+books(%rip)
	movl	$0, 308+books(%rip)
	movl	$0, 312+books(%rip)
	movl	$0, 316+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 320+books(%rip)
	movl	$0, 328+books(%rip)
	movl	$0, 332+books(%rip)
	movl	$0, 336+books(%rip)
	movl	$0, 340+books(%rip)
	movl	$0, 344+books(%rip)
	movl	$0, 348+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 352+books(%rip)
	movl	$0, 360+books(%rip)
	movl	$0, 364+books(%rip)
	movl	$0, 368+books(%rip)
	movl	$0, 372+books(%rip)
	movl	$0, 376+books(%rip)
	movl	$0, 380+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 384+books(%rip)
	movl	$0, 392+books(%rip)
	movl	$0, 396+books(%rip)
	movl	$0, 400+books(%rip)
	movl	$0, 404+books(%rip)
	movl	$0, 408+books(%rip)
	movl	$0, 412+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 416+books(%rip)
	movl	$0, 424+books(%rip)
	movl	$0, 428+books(%rip)
	movl	$0, 432+books(%rip)
	movl	$0, 436+books(%rip)
	movl	$0, 440+books(%rip)
	movl	$0, 444+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 448+books(%rip)
	movl	$0, 456+books(%rip)
	movl	$0, 460+books(%rip)
	movl	$0, 464+books(%rip)
	movl	$0, 468+books(%rip)
	movl	$0, 472+books(%rip)
	movl	$0, 476+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 480+books(%rip)
	movl	$0, 488+books(%rip)
	movl	$0, 492+books(%rip)
	movl	$0, 496+books(%rip)
	movl	$0, 500+books(%rip)
	movl	$0, 504+books(%rip)
	movl	$0, 508+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 512+books(%rip)
	movl	$0, 520+books(%rip)
	movl	$0, 524+books(%rip)
	movl	$0, 528+books(%rip)
	movl	$0, 532+books(%rip)
	movl	$0, 536+books(%rip)
	movl	$0, 540+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 544+books(%rip)
	movl	$0, 552+books(%rip)
	movl	$0, 556+books(%rip)
	movl	$0, 560+books(%rip)
	movl	$0, 564+books(%rip)
	movl	$0, 568+books(%rip)
	movl	$0, 572+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 576+books(%rip)
	movl	$0, 584+books(%rip)
	movl	$0, 588+books(%rip)
	movl	$0, 592+books(%rip)
	movl	$0, 596+books(%rip)
	movl	$0, 600+books(%rip)
	movl	$0, 604+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 608+books(%rip)
	movl	$0, 616+books(%rip)
	movl	$0, 620+books(%rip)
	movl	$0, 624+books(%rip)
	movl	$0, 628+books(%rip)
	movl	$0, 632+books(%rip)
	movl	$0, 636+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 640+books(%rip)
	movl	$0, 648+books(%rip)
	movl	$0, 652+books(%rip)
	movl	$0, 656+books(%rip)
	movl	$0, 660+books(%rip)
	movl	$0, 664+books(%rip)
	movl	$0, 668+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 672+books(%rip)
	movl	$0, 680+books(%rip)
	movl	$0, 684+books(%rip)
	movl	$0, 688+books(%rip)
	movl	$0, 692+books(%rip)
	movl	$0, 696+books(%rip)
	movl	$0, 700+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 704+books(%rip)
	movl	$0, 712+books(%rip)
	movl	$0, 716+books(%rip)
	movl	$0, 720+books(%rip)
	movl	$0, 724+books(%rip)
	movl	$0, 728+books(%rip)
	movl	$0, 732+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 736+books(%rip)
	movl	$0, 744+books(%rip)
	movl	$0, 748+books(%rip)
	movl	$0, 752+books(%rip)
	movl	$0, 756+books(%rip)
	movl	$0, 760+books(%rip)
	movl	$0, 764+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 768+books(%rip)
	movl	$0, 776+books(%rip)
	movl	$0, 780+books(%rip)
	movl	$0, 784+books(%rip)
	movl	$0, 788+books(%rip)
	movl	$0, 792+books(%rip)
	movl	$0, 796+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 800+books(%rip)
	movl	$0, 808+books(%rip)
	movl	$0, 812+books(%rip)
	movl	$0, 816+books(%rip)
	movl	$0, 820+books(%rip)
	movl	$0, 824+books(%rip)
	movl	$0, 828+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 832+books(%rip)
	movl	$0, 840+books(%rip)
	movl	$0, 844+books(%rip)
	movl	$0, 848+books(%rip)
	movl	$0, 852+books(%rip)
	movl	$0, 856+books(%rip)
	movl	$0, 860+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 864+books(%rip)
	movl	$0, 872+books(%rip)
	movl	$0, 876+books(%rip)
	movl	$0, 880+books(%rip)
	movl	$0, 884+books(%rip)
	movl	$0, 888+books(%rip)
	movl	$0, 892+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 896+books(%rip)
	movl	$0, 904+books(%rip)
	movl	$0, 908+books(%rip)
	movl	$0, 912+books(%rip)
	movl	$0, 916+books(%rip)
	movl	$0, 920+books(%rip)
	movl	$0, 924+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 928+books(%rip)
	movl	$0, 936+books(%rip)
	movl	$0, 940+books(%rip)
	movl	$0, 944+books(%rip)
	movl	$0, 948+books(%rip)
	movl	$0, 952+books(%rip)
	movl	$0, 956+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 960+books(%rip)
	movl	$0, 968+books(%rip)
	movl	$0, 972+books(%rip)
	movl	$0, 976+books(%rip)
	movl	$0, 980+books(%rip)
	movl	$0, 984+books(%rip)
	movl	$0, 988+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 992+books(%rip)
	movl	$0, 1000+books(%rip)
	movl	$0, 1004+books(%rip)
	movl	$0, 1008+books(%rip)
	movl	$0, 1012+books(%rip)
	movl	$0, 1016+books(%rip)
	movl	$0, 1020+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1024+books(%rip)
	movl	$0, 1032+books(%rip)
	movl	$0, 1036+books(%rip)
	movl	$0, 1040+books(%rip)
	movl	$0, 1044+books(%rip)
	movl	$0, 1048+books(%rip)
	movl	$0, 1052+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1056+books(%rip)
	movl	$0, 1064+books(%rip)
	movl	$0, 1068+books(%rip)
	movl	$0, 1072+books(%rip)
	movl	$0, 1076+books(%rip)
	movl	$0, 1080+books(%rip)
	movl	$0, 1084+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1088+books(%rip)
	movl	$0, 1096+books(%rip)
	movl	$0, 1100+books(%rip)
	movl	$0, 1104+books(%rip)
	movl	$0, 1108+books(%rip)
	movl	$0, 1112+books(%rip)
	movl	$0, 1116+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1120+books(%rip)
	movl	$0, 1128+books(%rip)
	movl	$0, 1132+books(%rip)
	movl	$0, 1136+books(%rip)
	movl	$0, 1140+books(%rip)
	movl	$0, 1144+books(%rip)
	movl	$0, 1148+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1152+books(%rip)
	movl	$0, 1160+books(%rip)
	movl	$0, 1164+books(%rip)
	movl	$0, 1168+books(%rip)
	movl	$0, 1172+books(%rip)
	movl	$0, 1176+books(%rip)
	movl	$0, 1180+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1184+books(%rip)
	movl	$0, 1192+books(%rip)
	movl	$0, 1196+books(%rip)
	movl	$0, 1200+books(%rip)
	movl	$0, 1204+books(%rip)
	movl	$0, 1208+books(%rip)
	movl	$0, 1212+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1216+books(%rip)
	movl	$0, 1224+books(%rip)
	movl	$0, 1228+books(%rip)
	movl	$0, 1232+books(%rip)
	movl	$0, 1236+books(%rip)
	movl	$0, 1240+books(%rip)
	movl	$0, 1244+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1248+books(%rip)
	movl	$0, 1256+books(%rip)
	movl	$0, 1260+books(%rip)
	movl	$0, 1264+books(%rip)
	movl	$0, 1268+books(%rip)
	movl	$0, 1272+books(%rip)
	movl	$0, 1276+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1280+books(%rip)
	movl	$0, 1288+books(%rip)
	movl	$0, 1292+books(%rip)
	movl	$0, 1296+books(%rip)
	movl	$0, 1300+books(%rip)
	movl	$0, 1304+books(%rip)
	movl	$0, 1308+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1312+books(%rip)
	movl	$0, 1320+books(%rip)
	movl	$0, 1324+books(%rip)
	movl	$0, 1328+books(%rip)
	movl	$0, 1332+books(%rip)
	movl	$0, 1336+books(%rip)
	movl	$0, 1340+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1344+books(%rip)
	movl	$0, 1352+books(%rip)
	movl	$0, 1356+books(%rip)
	movl	$0, 1360+books(%rip)
	movl	$0, 1364+books(%rip)
	movl	$0, 1368+books(%rip)
	movl	$0, 1372+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1376+books(%rip)
	movl	$0, 1384+books(%rip)
	movl	$0, 1388+books(%rip)
	movl	$0, 1392+books(%rip)
	movl	$0, 1396+books(%rip)
	movl	$0, 1400+books(%rip)
	movl	$0, 1404+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1408+books(%rip)
	movl	$0, 1416+books(%rip)
	movl	$0, 1420+books(%rip)
	movl	$0, 1424+books(%rip)
	movl	$0, 1428+books(%rip)
	movl	$0, 1432+books(%rip)
	movl	$0, 1436+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1440+books(%rip)
	movl	$0, 1448+books(%rip)
	movl	$0, 1452+books(%rip)
	movl	$0, 1456+books(%rip)
	movl	$0, 1460+books(%rip)
	movl	$0, 1464+books(%rip)
	movl	$0, 1468+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1472+books(%rip)
	movl	$0, 1480+books(%rip)
	movl	$0, 1484+books(%rip)
	movl	$0, 1488+books(%rip)
	movl	$0, 1492+books(%rip)
	movl	$0, 1496+books(%rip)
	movl	$0, 1500+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1504+books(%rip)
	movl	$0, 1512+books(%rip)
	movl	$0, 1516+books(%rip)
	movl	$0, 1520+books(%rip)
	movl	$0, 1524+books(%rip)
	movl	$0, 1528+books(%rip)
	movl	$0, 1532+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1536+books(%rip)
	movl	$0, 1544+books(%rip)
	movl	$0, 1548+books(%rip)
	movl	$0, 1552+books(%rip)
	movl	$0, 1556+books(%rip)
	movl	$0, 1560+books(%rip)
	movl	$0, 1564+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1568+books(%rip)
	movl	$0, 1576+books(%rip)
	movl	$0, 1580+books(%rip)
	movl	$0, 1584+books(%rip)
	movl	$0, 1588+books(%rip)
	movl	$0, 1592+books(%rip)
	movl	$0, 1596+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1600+books(%rip)
	movl	$0, 1608+books(%rip)
	movl	$0, 1612+books(%rip)
	movl	$0, 1616+books(%rip)
	movl	$0, 1620+books(%rip)
	movl	$0, 1624+books(%rip)
	movl	$0, 1628+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1632+books(%rip)
	movl	$0, 1640+books(%rip)
	movl	$0, 1644+books(%rip)
	movl	$0, 1648+books(%rip)
	movl	$0, 1652+books(%rip)
	movl	$0, 1656+books(%rip)
	movl	$0, 1660+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1664+books(%rip)
	movl	$0, 1672+books(%rip)
	movl	$0, 1676+books(%rip)
	movl	$0, 1680+books(%rip)
	movl	$0, 1684+books(%rip)
	movl	$0, 1688+books(%rip)
	movl	$0, 1692+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1696+books(%rip)
	movl	$0, 1704+books(%rip)
	movl	$0, 1708+books(%rip)
	movl	$0, 1712+books(%rip)
	movl	$0, 1716+books(%rip)
	movl	$0, 1720+books(%rip)
	movl	$0, 1724+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1728+books(%rip)
	movl	$0, 1736+books(%rip)
	movl	$0, 1740+books(%rip)
	movl	$0, 1744+books(%rip)
	movl	$0, 1748+books(%rip)
	movl	$0, 1752+books(%rip)
	movl	$0, 1756+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1760+books(%rip)
	movl	$0, 1768+books(%rip)
	movl	$0, 1772+books(%rip)
	movl	$0, 1776+books(%rip)
	movl	$0, 1780+books(%rip)
	movl	$0, 1784+books(%rip)
	movl	$0, 1788+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1792+books(%rip)
	movl	$0, 1800+books(%rip)
	movl	$0, 1804+books(%rip)
	movl	$0, 1808+books(%rip)
	movl	$0, 1812+books(%rip)
	movl	$0, 1816+books(%rip)
	movl	$0, 1820+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1824+books(%rip)
	movl	$0, 1832+books(%rip)
	movl	$0, 1836+books(%rip)
	movl	$0, 1840+books(%rip)
	movl	$0, 1844+books(%rip)
	movl	$0, 1848+books(%rip)
	movl	$0, 1852+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1856+books(%rip)
	movl	$0, 1864+books(%rip)
	movl	$0, 1868+books(%rip)
	movl	$0, 1872+books(%rip)
	movl	$0, 1876+books(%rip)
	movl	$0, 1880+books(%rip)
	movl	$0, 1884+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1888+books(%rip)
	movl	$0, 1896+books(%rip)
	movl	$0, 1900+books(%rip)
	movl	$0, 1904+books(%rip)
	movl	$0, 1908+books(%rip)
	movl	$0, 1912+books(%rip)
	movl	$0, 1916+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1920+books(%rip)
	movl	$0, 1928+books(%rip)
	movl	$0, 1932+books(%rip)
	movl	$0, 1936+books(%rip)
	movl	$0, 1940+books(%rip)
	movl	$0, 1944+books(%rip)
	movl	$0, 1948+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1952+books(%rip)
	movl	$0, 1960+books(%rip)
	movl	$0, 1964+books(%rip)
	movl	$0, 1968+books(%rip)
	movl	$0, 1972+books(%rip)
	movl	$0, 1976+books(%rip)
	movl	$0, 1980+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 1984+books(%rip)
	movl	$0, 1992+books(%rip)
	movl	$0, 1996+books(%rip)
	movl	$0, 2000+books(%rip)
	movl	$0, 2004+books(%rip)
	movl	$0, 2008+books(%rip)
	movl	$0, 2012+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2016+books(%rip)
	movl	$0, 2024+books(%rip)
	movl	$0, 2028+books(%rip)
	movl	$0, 2032+books(%rip)
	movl	$0, 2036+books(%rip)
	movl	$0, 2040+books(%rip)
	movl	$0, 2044+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2048+books(%rip)
	movl	$0, 2056+books(%rip)
	movl	$0, 2060+books(%rip)
	movl	$0, 2064+books(%rip)
	movl	$0, 2068+books(%rip)
	movl	$0, 2072+books(%rip)
	movl	$0, 2076+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2080+books(%rip)
	movl	$0, 2088+books(%rip)
	movl	$0, 2092+books(%rip)
	movl	$0, 2096+books(%rip)
	movl	$0, 2100+books(%rip)
	movl	$0, 2104+books(%rip)
	movl	$0, 2108+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2112+books(%rip)
	movl	$0, 2120+books(%rip)
	movl	$0, 2124+books(%rip)
	movl	$0, 2128+books(%rip)
	movl	$0, 2132+books(%rip)
	movl	$0, 2136+books(%rip)
	movl	$0, 2140+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2144+books(%rip)
	movl	$0, 2152+books(%rip)
	movl	$0, 2156+books(%rip)
	movl	$0, 2160+books(%rip)
	movl	$0, 2164+books(%rip)
	movl	$0, 2168+books(%rip)
	movl	$0, 2172+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2176+books(%rip)
	movl	$0, 2184+books(%rip)
	movl	$0, 2188+books(%rip)
	movl	$0, 2192+books(%rip)
	movl	$0, 2196+books(%rip)
	movl	$0, 2200+books(%rip)
	movl	$0, 2204+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2208+books(%rip)
	movl	$0, 2216+books(%rip)
	movl	$0, 2220+books(%rip)
	movl	$0, 2224+books(%rip)
	movl	$0, 2228+books(%rip)
	movl	$0, 2232+books(%rip)
	movl	$0, 2236+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2240+books(%rip)
	movl	$0, 2248+books(%rip)
	movl	$0, 2252+books(%rip)
	movl	$0, 2256+books(%rip)
	movl	$0, 2260+books(%rip)
	movl	$0, 2264+books(%rip)
	movl	$0, 2268+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2272+books(%rip)
	movl	$0, 2280+books(%rip)
	movl	$0, 2284+books(%rip)
	movl	$0, 2288+books(%rip)
	movl	$0, 2292+books(%rip)
	movl	$0, 2296+books(%rip)
	movl	$0, 2300+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2304+books(%rip)
	movl	$0, 2312+books(%rip)
	movl	$0, 2316+books(%rip)
	movl	$0, 2320+books(%rip)
	movl	$0, 2324+books(%rip)
	movl	$0, 2328+books(%rip)
	movl	$0, 2332+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2336+books(%rip)
	movl	$0, 2344+books(%rip)
	movl	$0, 2348+books(%rip)
	movl	$0, 2352+books(%rip)
	movl	$0, 2356+books(%rip)
	movl	$0, 2360+books(%rip)
	movl	$0, 2364+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2368+books(%rip)
	movl	$0, 2376+books(%rip)
	movl	$0, 2380+books(%rip)
	movl	$0, 2384+books(%rip)
	movl	$0, 2388+books(%rip)
	movl	$0, 2392+books(%rip)
	movl	$0, 2396+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2400+books(%rip)
	movl	$0, 2408+books(%rip)
	movl	$0, 2412+books(%rip)
	movl	$0, 2416+books(%rip)
	movl	$0, 2420+books(%rip)
	movl	$0, 2424+books(%rip)
	movl	$0, 2428+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2432+books(%rip)
	movl	$0, 2440+books(%rip)
	movl	$0, 2444+books(%rip)
	movl	$0, 2448+books(%rip)
	movl	$0, 2452+books(%rip)
	movl	$0, 2456+books(%rip)
	movl	$0, 2460+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2464+books(%rip)
	movl	$0, 2472+books(%rip)
	movl	$0, 2476+books(%rip)
	movl	$0, 2480+books(%rip)
	movl	$0, 2484+books(%rip)
	movl	$0, 2488+books(%rip)
	movl	$0, 2492+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2496+books(%rip)
	movl	$0, 2504+books(%rip)
	movl	$0, 2508+books(%rip)
	movl	$0, 2512+books(%rip)
	movl	$0, 2516+books(%rip)
	movl	$0, 2520+books(%rip)
	movl	$0, 2524+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2528+books(%rip)
	movl	$0, 2536+books(%rip)
	movl	$0, 2540+books(%rip)
	movl	$0, 2544+books(%rip)
	movl	$0, 2548+books(%rip)
	movl	$0, 2552+books(%rip)
	movl	$0, 2556+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2560+books(%rip)
	movl	$0, 2568+books(%rip)
	movl	$0, 2572+books(%rip)
	movl	$0, 2576+books(%rip)
	movl	$0, 2580+books(%rip)
	movl	$0, 2584+books(%rip)
	movl	$0, 2588+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2592+books(%rip)
	movl	$0, 2600+books(%rip)
	movl	$0, 2604+books(%rip)
	movl	$0, 2608+books(%rip)
	movl	$0, 2612+books(%rip)
	movl	$0, 2616+books(%rip)
	movl	$0, 2620+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2624+books(%rip)
	movl	$0, 2632+books(%rip)
	movl	$0, 2636+books(%rip)
	movl	$0, 2640+books(%rip)
	movl	$0, 2644+books(%rip)
	movl	$0, 2648+books(%rip)
	movl	$0, 2652+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2656+books(%rip)
	movl	$0, 2664+books(%rip)
	movl	$0, 2668+books(%rip)
	movl	$0, 2672+books(%rip)
	movl	$0, 2676+books(%rip)
	movl	$0, 2680+books(%rip)
	movl	$0, 2684+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2688+books(%rip)
	movl	$0, 2696+books(%rip)
	movl	$0, 2700+books(%rip)
	movl	$0, 2704+books(%rip)
	movl	$0, 2708+books(%rip)
	movl	$0, 2712+books(%rip)
	movl	$0, 2716+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2720+books(%rip)
	movl	$0, 2728+books(%rip)
	movl	$0, 2732+books(%rip)
	movl	$0, 2736+books(%rip)
	movl	$0, 2740+books(%rip)
	movl	$0, 2744+books(%rip)
	movl	$0, 2748+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2752+books(%rip)
	movl	$0, 2760+books(%rip)
	movl	$0, 2764+books(%rip)
	movl	$0, 2768+books(%rip)
	movl	$0, 2772+books(%rip)
	movl	$0, 2776+books(%rip)
	movl	$0, 2780+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2784+books(%rip)
	movl	$0, 2792+books(%rip)
	movl	$0, 2796+books(%rip)
	movl	$0, 2800+books(%rip)
	movl	$0, 2804+books(%rip)
	movl	$0, 2808+books(%rip)
	movl	$0, 2812+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2816+books(%rip)
	movl	$0, 2824+books(%rip)
	movl	$0, 2828+books(%rip)
	movl	$0, 2832+books(%rip)
	movl	$0, 2836+books(%rip)
	movl	$0, 2840+books(%rip)
	movl	$0, 2844+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2848+books(%rip)
	movl	$0, 2856+books(%rip)
	movl	$0, 2860+books(%rip)
	movl	$0, 2864+books(%rip)
	movl	$0, 2868+books(%rip)
	movl	$0, 2872+books(%rip)
	movl	$0, 2876+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2880+books(%rip)
	movl	$0, 2888+books(%rip)
	movl	$0, 2892+books(%rip)
	movl	$0, 2896+books(%rip)
	movl	$0, 2900+books(%rip)
	movl	$0, 2904+books(%rip)
	movl	$0, 2908+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2912+books(%rip)
	movl	$0, 2920+books(%rip)
	movl	$0, 2924+books(%rip)
	movl	$0, 2928+books(%rip)
	movl	$0, 2932+books(%rip)
	movl	$0, 2936+books(%rip)
	movl	$0, 2940+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2944+books(%rip)
	movl	$0, 2952+books(%rip)
	movl	$0, 2956+books(%rip)
	movl	$0, 2960+books(%rip)
	movl	$0, 2964+books(%rip)
	movl	$0, 2968+books(%rip)
	movl	$0, 2972+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 2976+books(%rip)
	movl	$0, 2984+books(%rip)
	movl	$0, 2988+books(%rip)
	movl	$0, 2992+books(%rip)
	movl	$0, 2996+books(%rip)
	movl	$0, 3000+books(%rip)
	movl	$0, 3004+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3008+books(%rip)
	movl	$0, 3016+books(%rip)
	movl	$0, 3020+books(%rip)
	movl	$0, 3024+books(%rip)
	movl	$0, 3028+books(%rip)
	movl	$0, 3032+books(%rip)
	movl	$0, 3036+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3040+books(%rip)
	movl	$0, 3048+books(%rip)
	movl	$0, 3052+books(%rip)
	movl	$0, 3056+books(%rip)
	movl	$0, 3060+books(%rip)
	movl	$0, 3064+books(%rip)
	movl	$0, 3068+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3072+books(%rip)
	movl	$0, 3080+books(%rip)
	movl	$0, 3084+books(%rip)
	movl	$0, 3088+books(%rip)
	movl	$0, 3092+books(%rip)
	movl	$0, 3096+books(%rip)
	movl	$0, 3100+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3104+books(%rip)
	movl	$0, 3112+books(%rip)
	movl	$0, 3116+books(%rip)
	movl	$0, 3120+books(%rip)
	movl	$0, 3124+books(%rip)
	movl	$0, 3128+books(%rip)
	movl	$0, 3132+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3136+books(%rip)
	movl	$0, 3144+books(%rip)
	movl	$0, 3148+books(%rip)
	movl	$0, 3152+books(%rip)
	movl	$0, 3156+books(%rip)
	movl	$0, 3160+books(%rip)
	movl	$0, 3164+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3168+books(%rip)
	movl	$0, 3176+books(%rip)
	movl	$0, 3180+books(%rip)
	movl	$0, 3184+books(%rip)
	movl	$0, 3188+books(%rip)
	movl	$0, 3192+books(%rip)
	movl	$0, 3196+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3200+books(%rip)
	movl	$0, 3208+books(%rip)
	movl	$0, 3212+books(%rip)
	movl	$0, 3216+books(%rip)
	movl	$0, 3220+books(%rip)
	movl	$0, 3224+books(%rip)
	movl	$0, 3228+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3232+books(%rip)
	movl	$0, 3240+books(%rip)
	movl	$0, 3244+books(%rip)
	movl	$0, 3248+books(%rip)
	movl	$0, 3252+books(%rip)
	movl	$0, 3256+books(%rip)
	movl	$0, 3260+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3264+books(%rip)
	movl	$0, 3272+books(%rip)
	movl	$0, 3276+books(%rip)
	movl	$0, 3280+books(%rip)
	movl	$0, 3284+books(%rip)
	movl	$0, 3288+books(%rip)
	movl	$0, 3292+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3296+books(%rip)
	movl	$0, 3304+books(%rip)
	movl	$0, 3308+books(%rip)
	movl	$0, 3312+books(%rip)
	movl	$0, 3316+books(%rip)
	movl	$0, 3320+books(%rip)
	movl	$0, 3324+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3328+books(%rip)
	movl	$0, 3336+books(%rip)
	movl	$0, 3340+books(%rip)
	movl	$0, 3344+books(%rip)
	movl	$0, 3348+books(%rip)
	movl	$0, 3352+books(%rip)
	movl	$0, 3356+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3360+books(%rip)
	movl	$0, 3368+books(%rip)
	movl	$0, 3372+books(%rip)
	movl	$0, 3376+books(%rip)
	movl	$0, 3380+books(%rip)
	movl	$0, 3384+books(%rip)
	movl	$0, 3388+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3392+books(%rip)
	movl	$0, 3400+books(%rip)
	movl	$0, 3404+books(%rip)
	movl	$0, 3408+books(%rip)
	movl	$0, 3412+books(%rip)
	movl	$0, 3416+books(%rip)
	movl	$0, 3420+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3424+books(%rip)
	movl	$0, 3432+books(%rip)
	movl	$0, 3436+books(%rip)
	movl	$0, 3440+books(%rip)
	movl	$0, 3444+books(%rip)
	movl	$0, 3448+books(%rip)
	movl	$0, 3452+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3456+books(%rip)
	movl	$0, 3464+books(%rip)
	movl	$0, 3468+books(%rip)
	movl	$0, 3472+books(%rip)
	movl	$0, 3476+books(%rip)
	movl	$0, 3480+books(%rip)
	movl	$0, 3484+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3488+books(%rip)
	movl	$0, 3496+books(%rip)
	movl	$0, 3500+books(%rip)
	movl	$0, 3504+books(%rip)
	movl	$0, 3508+books(%rip)
	movl	$0, 3512+books(%rip)
	movl	$0, 3516+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3520+books(%rip)
	movl	$0, 3528+books(%rip)
	movl	$0, 3532+books(%rip)
	movl	$0, 3536+books(%rip)
	movl	$0, 3540+books(%rip)
	movl	$0, 3544+books(%rip)
	movl	$0, 3548+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3552+books(%rip)
	movl	$0, 3560+books(%rip)
	movl	$0, 3564+books(%rip)
	movl	$0, 3568+books(%rip)
	movl	$0, 3572+books(%rip)
	movl	$0, 3576+books(%rip)
	movl	$0, 3580+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3584+books(%rip)
	movl	$0, 3592+books(%rip)
	movl	$0, 3596+books(%rip)
	movl	$0, 3600+books(%rip)
	movl	$0, 3604+books(%rip)
	movl	$0, 3608+books(%rip)
	movl	$0, 3612+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3616+books(%rip)
	movl	$0, 3624+books(%rip)
	movl	$0, 3628+books(%rip)
	movl	$0, 3632+books(%rip)
	movl	$0, 3636+books(%rip)
	movl	$0, 3640+books(%rip)
	movl	$0, 3644+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3648+books(%rip)
	movl	$0, 3656+books(%rip)
	movl	$0, 3660+books(%rip)
	movl	$0, 3664+books(%rip)
	movl	$0, 3668+books(%rip)
	movl	$0, 3672+books(%rip)
	movl	$0, 3676+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3680+books(%rip)
	movl	$0, 3688+books(%rip)
	movl	$0, 3692+books(%rip)
	movl	$0, 3696+books(%rip)
	movl	$0, 3700+books(%rip)
	movl	$0, 3704+books(%rip)
	movl	$0, 3708+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3712+books(%rip)
	movl	$0, 3720+books(%rip)
	movl	$0, 3724+books(%rip)
	movl	$0, 3728+books(%rip)
	movl	$0, 3732+books(%rip)
	movl	$0, 3736+books(%rip)
	movl	$0, 3740+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3744+books(%rip)
	movl	$0, 3752+books(%rip)
	movl	$0, 3756+books(%rip)
	movl	$0, 3760+books(%rip)
	movl	$0, 3764+books(%rip)
	movl	$0, 3768+books(%rip)
	movl	$0, 3772+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3776+books(%rip)
	movl	$0, 3784+books(%rip)
	movl	$0, 3788+books(%rip)
	movl	$0, 3792+books(%rip)
	movl	$0, 3796+books(%rip)
	movl	$0, 3800+books(%rip)
	movl	$0, 3804+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3808+books(%rip)
	movl	$0, 3816+books(%rip)
	movl	$0, 3820+books(%rip)
	movl	$0, 3824+books(%rip)
	movl	$0, 3828+books(%rip)
	movl	$0, 3832+books(%rip)
	movl	$0, 3836+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3840+books(%rip)
	movl	$0, 3848+books(%rip)
	movl	$0, 3852+books(%rip)
	movl	$0, 3856+books(%rip)
	movl	$0, 3860+books(%rip)
	movl	$0, 3864+books(%rip)
	movl	$0, 3868+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3872+books(%rip)
	movl	$0, 3880+books(%rip)
	movl	$0, 3884+books(%rip)
	movl	$0, 3888+books(%rip)
	movl	$0, 3892+books(%rip)
	movl	$0, 3896+books(%rip)
	movl	$0, 3900+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3904+books(%rip)
	movl	$0, 3912+books(%rip)
	movl	$0, 3916+books(%rip)
	movl	$0, 3920+books(%rip)
	movl	$0, 3924+books(%rip)
	movl	$0, 3928+books(%rip)
	movl	$0, 3932+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3936+books(%rip)
	movl	$0, 3944+books(%rip)
	movl	$0, 3948+books(%rip)
	movl	$0, 3952+books(%rip)
	movl	$0, 3956+books(%rip)
	movl	$0, 3960+books(%rip)
	movl	$0, 3964+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 3968+books(%rip)
	movl	$0, 3976+books(%rip)
	movl	$0, 3980+books(%rip)
	movl	$0, 3984+books(%rip)
	movl	$0, 3988+books(%rip)
	movl	$0, 3992+books(%rip)
	movl	$0, 3996+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4000+books(%rip)
	movl	$0, 4008+books(%rip)
	movl	$0, 4012+books(%rip)
	movl	$0, 4016+books(%rip)
	movl	$0, 4020+books(%rip)
	movl	$0, 4024+books(%rip)
	movl	$0, 4028+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4032+books(%rip)
	movl	$0, 4040+books(%rip)
	movl	$0, 4044+books(%rip)
	movl	$0, 4048+books(%rip)
	movl	$0, 4052+books(%rip)
	movl	$0, 4056+books(%rip)
	movl	$0, 4060+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4064+books(%rip)
	movl	$0, 4072+books(%rip)
	movl	$0, 4076+books(%rip)
	movl	$0, 4080+books(%rip)
	movl	$0, 4084+books(%rip)
	movl	$0, 4088+books(%rip)
	movl	$0, 4092+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4096+books(%rip)
	movl	$0, 4104+books(%rip)
	movl	$0, 4108+books(%rip)
	movl	$0, 4112+books(%rip)
	movl	$0, 4116+books(%rip)
	movl	$0, 4120+books(%rip)
	movl	$0, 4124+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4128+books(%rip)
	movl	$0, 4136+books(%rip)
	movl	$0, 4140+books(%rip)
	movl	$0, 4144+books(%rip)
	movl	$0, 4148+books(%rip)
	movl	$0, 4152+books(%rip)
	movl	$0, 4156+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4160+books(%rip)
	movl	$0, 4168+books(%rip)
	movl	$0, 4172+books(%rip)
	movl	$0, 4176+books(%rip)
	movl	$0, 4180+books(%rip)
	movl	$0, 4184+books(%rip)
	movl	$0, 4188+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4192+books(%rip)
	movl	$0, 4200+books(%rip)
	movl	$0, 4204+books(%rip)
	movl	$0, 4208+books(%rip)
	movl	$0, 4212+books(%rip)
	movl	$0, 4216+books(%rip)
	movl	$0, 4220+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4224+books(%rip)
	movl	$0, 4232+books(%rip)
	movl	$0, 4236+books(%rip)
	movl	$0, 4240+books(%rip)
	movl	$0, 4244+books(%rip)
	movl	$0, 4248+books(%rip)
	movl	$0, 4252+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4256+books(%rip)
	movl	$0, 4264+books(%rip)
	movl	$0, 4268+books(%rip)
	movl	$0, 4272+books(%rip)
	movl	$0, 4276+books(%rip)
	movl	$0, 4280+books(%rip)
	movl	$0, 4284+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4288+books(%rip)
	movl	$0, 4296+books(%rip)
	movl	$0, 4300+books(%rip)
	movl	$0, 4304+books(%rip)
	movl	$0, 4308+books(%rip)
	movl	$0, 4312+books(%rip)
	movl	$0, 4316+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4320+books(%rip)
	movl	$0, 4328+books(%rip)
	movl	$0, 4332+books(%rip)
	movl	$0, 4336+books(%rip)
	movl	$0, 4340+books(%rip)
	movl	$0, 4344+books(%rip)
	movl	$0, 4348+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4352+books(%rip)
	movl	$0, 4360+books(%rip)
	movl	$0, 4364+books(%rip)
	movl	$0, 4368+books(%rip)
	movl	$0, 4372+books(%rip)
	movl	$0, 4376+books(%rip)
	movl	$0, 4380+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4384+books(%rip)
	movl	$0, 4392+books(%rip)
	movl	$0, 4396+books(%rip)
	movl	$0, 4400+books(%rip)
	movl	$0, 4404+books(%rip)
	movl	$0, 4408+books(%rip)
	movl	$0, 4412+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4416+books(%rip)
	movl	$0, 4424+books(%rip)
	movl	$0, 4428+books(%rip)
	movl	$0, 4432+books(%rip)
	movl	$0, 4436+books(%rip)
	movl	$0, 4440+books(%rip)
	movl	$0, 4444+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4448+books(%rip)
	movl	$0, 4456+books(%rip)
	movl	$0, 4460+books(%rip)
	movl	$0, 4464+books(%rip)
	movl	$0, 4468+books(%rip)
	movl	$0, 4472+books(%rip)
	movl	$0, 4476+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4480+books(%rip)
	movl	$0, 4488+books(%rip)
	movl	$0, 4492+books(%rip)
	movl	$0, 4496+books(%rip)
	movl	$0, 4500+books(%rip)
	movl	$0, 4504+books(%rip)
	movl	$0, 4508+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4512+books(%rip)
	movl	$0, 4520+books(%rip)
	movl	$0, 4524+books(%rip)
	movl	$0, 4528+books(%rip)
	movl	$0, 4532+books(%rip)
	movl	$0, 4536+books(%rip)
	movl	$0, 4540+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4544+books(%rip)
	movl	$0, 4552+books(%rip)
	movl	$0, 4556+books(%rip)
	movl	$0, 4560+books(%rip)
	movl	$0, 4564+books(%rip)
	movl	$0, 4568+books(%rip)
	movl	$0, 4572+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4576+books(%rip)
	movl	$0, 4584+books(%rip)
	movl	$0, 4588+books(%rip)
	movl	$0, 4592+books(%rip)
	movl	$0, 4596+books(%rip)
	movl	$0, 4600+books(%rip)
	movl	$0, 4604+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4608+books(%rip)
	movl	$0, 4616+books(%rip)
	movl	$0, 4620+books(%rip)
	movl	$0, 4624+books(%rip)
	movl	$0, 4628+books(%rip)
	movl	$0, 4632+books(%rip)
	movl	$0, 4636+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4640+books(%rip)
	movl	$0, 4648+books(%rip)
	movl	$0, 4652+books(%rip)
	movl	$0, 4656+books(%rip)
	movl	$0, 4660+books(%rip)
	movl	$0, 4664+books(%rip)
	movl	$0, 4668+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4672+books(%rip)
	movl	$0, 4680+books(%rip)
	movl	$0, 4684+books(%rip)
	movl	$0, 4688+books(%rip)
	movl	$0, 4692+books(%rip)
	movl	$0, 4696+books(%rip)
	movl	$0, 4700+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4704+books(%rip)
	movl	$0, 4712+books(%rip)
	movl	$0, 4716+books(%rip)
	movl	$0, 4720+books(%rip)
	movl	$0, 4724+books(%rip)
	movl	$0, 4728+books(%rip)
	movl	$0, 4732+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4736+books(%rip)
	movl	$0, 4744+books(%rip)
	movl	$0, 4748+books(%rip)
	movl	$0, 4752+books(%rip)
	movl	$0, 4756+books(%rip)
	movl	$0, 4760+books(%rip)
	movl	$0, 4764+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4768+books(%rip)
	movl	$0, 4776+books(%rip)
	movl	$0, 4780+books(%rip)
	movl	$0, 4784+books(%rip)
	movl	$0, 4788+books(%rip)
	movl	$0, 4792+books(%rip)
	movl	$0, 4796+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4800+books(%rip)
	movl	$0, 4808+books(%rip)
	movl	$0, 4812+books(%rip)
	movl	$0, 4816+books(%rip)
	movl	$0, 4820+books(%rip)
	movl	$0, 4824+books(%rip)
	movl	$0, 4828+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4832+books(%rip)
	movl	$0, 4840+books(%rip)
	movl	$0, 4844+books(%rip)
	movl	$0, 4848+books(%rip)
	movl	$0, 4852+books(%rip)
	movl	$0, 4856+books(%rip)
	movl	$0, 4860+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4864+books(%rip)
	movl	$0, 4872+books(%rip)
	movl	$0, 4876+books(%rip)
	movl	$0, 4880+books(%rip)
	movl	$0, 4884+books(%rip)
	movl	$0, 4888+books(%rip)
	movl	$0, 4892+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4896+books(%rip)
	movl	$0, 4904+books(%rip)
	movl	$0, 4908+books(%rip)
	movl	$0, 4912+books(%rip)
	movl	$0, 4916+books(%rip)
	movl	$0, 4920+books(%rip)
	movl	$0, 4924+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4928+books(%rip)
	movl	$0, 4936+books(%rip)
	movl	$0, 4940+books(%rip)
	movl	$0, 4944+books(%rip)
	movl	$0, 4948+books(%rip)
	movl	$0, 4952+books(%rip)
	movl	$0, 4956+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4960+books(%rip)
	movl	$0, 4968+books(%rip)
	movl	$0, 4972+books(%rip)
	movl	$0, 4976+books(%rip)
	movl	$0, 4980+books(%rip)
	movl	$0, 4984+books(%rip)
	movl	$0, 4988+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 4992+books(%rip)
	movl	$0, 5000+books(%rip)
	movl	$0, 5004+books(%rip)
	movl	$0, 5008+books(%rip)
	movl	$0, 5012+books(%rip)
	movl	$0, 5016+books(%rip)
	movl	$0, 5020+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5024+books(%rip)
	movl	$0, 5032+books(%rip)
	movl	$0, 5036+books(%rip)
	movl	$0, 5040+books(%rip)
	movl	$0, 5044+books(%rip)
	movl	$0, 5048+books(%rip)
	movl	$0, 5052+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5056+books(%rip)
	movl	$0, 5064+books(%rip)
	movl	$0, 5068+books(%rip)
	movl	$0, 5072+books(%rip)
	movl	$0, 5076+books(%rip)
	movl	$0, 5080+books(%rip)
	movl	$0, 5084+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5088+books(%rip)
	movl	$0, 5096+books(%rip)
	movl	$0, 5100+books(%rip)
	movl	$0, 5104+books(%rip)
	movl	$0, 5108+books(%rip)
	movl	$0, 5112+books(%rip)
	movl	$0, 5116+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5120+books(%rip)
	movl	$0, 5128+books(%rip)
	movl	$0, 5132+books(%rip)
	movl	$0, 5136+books(%rip)
	movl	$0, 5140+books(%rip)
	movl	$0, 5144+books(%rip)
	movl	$0, 5148+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5152+books(%rip)
	movl	$0, 5160+books(%rip)
	movl	$0, 5164+books(%rip)
	movl	$0, 5168+books(%rip)
	movl	$0, 5172+books(%rip)
	movl	$0, 5176+books(%rip)
	movl	$0, 5180+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5184+books(%rip)
	movl	$0, 5192+books(%rip)
	movl	$0, 5196+books(%rip)
	movl	$0, 5200+books(%rip)
	movl	$0, 5204+books(%rip)
	movl	$0, 5208+books(%rip)
	movl	$0, 5212+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5216+books(%rip)
	movl	$0, 5224+books(%rip)
	movl	$0, 5228+books(%rip)
	movl	$0, 5232+books(%rip)
	movl	$0, 5236+books(%rip)
	movl	$0, 5240+books(%rip)
	movl	$0, 5244+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5248+books(%rip)
	movl	$0, 5256+books(%rip)
	movl	$0, 5260+books(%rip)
	movl	$0, 5264+books(%rip)
	movl	$0, 5268+books(%rip)
	movl	$0, 5272+books(%rip)
	movl	$0, 5276+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5280+books(%rip)
	movl	$0, 5288+books(%rip)
	movl	$0, 5292+books(%rip)
	movl	$0, 5296+books(%rip)
	movl	$0, 5300+books(%rip)
	movl	$0, 5304+books(%rip)
	movl	$0, 5308+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5312+books(%rip)
	movl	$0, 5320+books(%rip)
	movl	$0, 5324+books(%rip)
	movl	$0, 5328+books(%rip)
	movl	$0, 5332+books(%rip)
	movl	$0, 5336+books(%rip)
	movl	$0, 5340+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5344+books(%rip)
	movl	$0, 5352+books(%rip)
	movl	$0, 5356+books(%rip)
	movl	$0, 5360+books(%rip)
	movl	$0, 5364+books(%rip)
	movl	$0, 5368+books(%rip)
	movl	$0, 5372+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5376+books(%rip)
	movl	$0, 5384+books(%rip)
	movl	$0, 5388+books(%rip)
	movl	$0, 5392+books(%rip)
	movl	$0, 5396+books(%rip)
	movl	$0, 5400+books(%rip)
	movl	$0, 5404+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5408+books(%rip)
	movl	$0, 5416+books(%rip)
	movl	$0, 5420+books(%rip)
	movl	$0, 5424+books(%rip)
	movl	$0, 5428+books(%rip)
	movl	$0, 5432+books(%rip)
	movl	$0, 5436+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5440+books(%rip)
	movl	$0, 5448+books(%rip)
	movl	$0, 5452+books(%rip)
	movl	$0, 5456+books(%rip)
	movl	$0, 5460+books(%rip)
	movl	$0, 5464+books(%rip)
	movl	$0, 5468+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5472+books(%rip)
	movl	$0, 5480+books(%rip)
	movl	$0, 5484+books(%rip)
	movl	$0, 5488+books(%rip)
	movl	$0, 5492+books(%rip)
	movl	$0, 5496+books(%rip)
	movl	$0, 5500+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5504+books(%rip)
	movl	$0, 5512+books(%rip)
	movl	$0, 5516+books(%rip)
	movl	$0, 5520+books(%rip)
	movl	$0, 5524+books(%rip)
	movl	$0, 5528+books(%rip)
	movl	$0, 5532+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5536+books(%rip)
	movl	$0, 5544+books(%rip)
	movl	$0, 5548+books(%rip)
	movl	$0, 5552+books(%rip)
	movl	$0, 5556+books(%rip)
	movl	$0, 5560+books(%rip)
	movl	$0, 5564+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5568+books(%rip)
	movl	$0, 5576+books(%rip)
	movl	$0, 5580+books(%rip)
	movl	$0, 5584+books(%rip)
	movl	$0, 5588+books(%rip)
	movl	$0, 5592+books(%rip)
	movl	$0, 5596+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5600+books(%rip)
	movl	$0, 5608+books(%rip)
	movl	$0, 5612+books(%rip)
	movl	$0, 5616+books(%rip)
	movl	$0, 5620+books(%rip)
	movl	$0, 5624+books(%rip)
	movl	$0, 5628+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5632+books(%rip)
	movl	$0, 5640+books(%rip)
	movl	$0, 5644+books(%rip)
	movl	$0, 5648+books(%rip)
	movl	$0, 5652+books(%rip)
	movl	$0, 5656+books(%rip)
	movl	$0, 5660+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5664+books(%rip)
	movl	$0, 5672+books(%rip)
	movl	$0, 5676+books(%rip)
	movl	$0, 5680+books(%rip)
	movl	$0, 5684+books(%rip)
	movl	$0, 5688+books(%rip)
	movl	$0, 5692+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5696+books(%rip)
	movl	$0, 5704+books(%rip)
	movl	$0, 5708+books(%rip)
	movl	$0, 5712+books(%rip)
	movl	$0, 5716+books(%rip)
	movl	$0, 5720+books(%rip)
	movl	$0, 5724+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5728+books(%rip)
	movl	$0, 5736+books(%rip)
	movl	$0, 5740+books(%rip)
	movl	$0, 5744+books(%rip)
	movl	$0, 5748+books(%rip)
	movl	$0, 5752+books(%rip)
	movl	$0, 5756+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5760+books(%rip)
	movl	$0, 5768+books(%rip)
	movl	$0, 5772+books(%rip)
	movl	$0, 5776+books(%rip)
	movl	$0, 5780+books(%rip)
	movl	$0, 5784+books(%rip)
	movl	$0, 5788+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5792+books(%rip)
	movl	$0, 5800+books(%rip)
	movl	$0, 5804+books(%rip)
	movl	$0, 5808+books(%rip)
	movl	$0, 5812+books(%rip)
	movl	$0, 5816+books(%rip)
	movl	$0, 5820+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5824+books(%rip)
	movl	$0, 5832+books(%rip)
	movl	$0, 5836+books(%rip)
	movl	$0, 5840+books(%rip)
	movl	$0, 5844+books(%rip)
	movl	$0, 5848+books(%rip)
	movl	$0, 5852+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5856+books(%rip)
	movl	$0, 5864+books(%rip)
	movl	$0, 5868+books(%rip)
	movl	$0, 5872+books(%rip)
	movl	$0, 5876+books(%rip)
	movl	$0, 5880+books(%rip)
	movl	$0, 5884+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5888+books(%rip)
	movl	$0, 5896+books(%rip)
	movl	$0, 5900+books(%rip)
	movl	$0, 5904+books(%rip)
	movl	$0, 5908+books(%rip)
	movl	$0, 5912+books(%rip)
	movl	$0, 5916+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5920+books(%rip)
	movl	$0, 5928+books(%rip)
	movl	$0, 5932+books(%rip)
	movl	$0, 5936+books(%rip)
	movl	$0, 5940+books(%rip)
	movl	$0, 5944+books(%rip)
	movl	$0, 5948+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5952+books(%rip)
	movl	$0, 5960+books(%rip)
	movl	$0, 5964+books(%rip)
	movl	$0, 5968+books(%rip)
	movl	$0, 5972+books(%rip)
	movl	$0, 5976+books(%rip)
	movl	$0, 5980+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 5984+books(%rip)
	movl	$0, 5992+books(%rip)
	movl	$0, 5996+books(%rip)
	movl	$0, 6000+books(%rip)
	movl	$0, 6004+books(%rip)
	movl	$0, 6008+books(%rip)
	movl	$0, 6012+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6016+books(%rip)
	movl	$0, 6024+books(%rip)
	movl	$0, 6028+books(%rip)
	movl	$0, 6032+books(%rip)
	movl	$0, 6036+books(%rip)
	movl	$0, 6040+books(%rip)
	movl	$0, 6044+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6048+books(%rip)
	movl	$0, 6056+books(%rip)
	movl	$0, 6060+books(%rip)
	movl	$0, 6064+books(%rip)
	movl	$0, 6068+books(%rip)
	movl	$0, 6072+books(%rip)
	movl	$0, 6076+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6080+books(%rip)
	movl	$0, 6088+books(%rip)
	movl	$0, 6092+books(%rip)
	movl	$0, 6096+books(%rip)
	movl	$0, 6100+books(%rip)
	movl	$0, 6104+books(%rip)
	movl	$0, 6108+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6112+books(%rip)
	movl	$0, 6120+books(%rip)
	movl	$0, 6124+books(%rip)
	movl	$0, 6128+books(%rip)
	movl	$0, 6132+books(%rip)
	movl	$0, 6136+books(%rip)
	movl	$0, 6140+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6144+books(%rip)
	movl	$0, 6152+books(%rip)
	movl	$0, 6156+books(%rip)
	movl	$0, 6160+books(%rip)
	movl	$0, 6164+books(%rip)
	movl	$0, 6168+books(%rip)
	movl	$0, 6172+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6176+books(%rip)
	movl	$0, 6184+books(%rip)
	movl	$0, 6188+books(%rip)
	movl	$0, 6192+books(%rip)
	movl	$0, 6196+books(%rip)
	movl	$0, 6200+books(%rip)
	movl	$0, 6204+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6208+books(%rip)
	movl	$0, 6216+books(%rip)
	movl	$0, 6220+books(%rip)
	movl	$0, 6224+books(%rip)
	movl	$0, 6228+books(%rip)
	movl	$0, 6232+books(%rip)
	movl	$0, 6236+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6240+books(%rip)
	movl	$0, 6248+books(%rip)
	movl	$0, 6252+books(%rip)
	movl	$0, 6256+books(%rip)
	movl	$0, 6260+books(%rip)
	movl	$0, 6264+books(%rip)
	movl	$0, 6268+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6272+books(%rip)
	movl	$0, 6280+books(%rip)
	movl	$0, 6284+books(%rip)
	movl	$0, 6288+books(%rip)
	movl	$0, 6292+books(%rip)
	movl	$0, 6296+books(%rip)
	movl	$0, 6300+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6304+books(%rip)
	movl	$0, 6312+books(%rip)
	movl	$0, 6316+books(%rip)
	movl	$0, 6320+books(%rip)
	movl	$0, 6324+books(%rip)
	movl	$0, 6328+books(%rip)
	movl	$0, 6332+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6336+books(%rip)
	movl	$0, 6344+books(%rip)
	movl	$0, 6348+books(%rip)
	movl	$0, 6352+books(%rip)
	movl	$0, 6356+books(%rip)
	movl	$0, 6360+books(%rip)
	movl	$0, 6364+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6368+books(%rip)
	movl	$0, 6376+books(%rip)
	movl	$0, 6380+books(%rip)
	movl	$0, 6384+books(%rip)
	movl	$0, 6388+books(%rip)
	movl	$0, 6392+books(%rip)
	movl	$0, 6396+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6400+books(%rip)
	movl	$0, 6408+books(%rip)
	movl	$0, 6412+books(%rip)
	movl	$0, 6416+books(%rip)
	movl	$0, 6420+books(%rip)
	movl	$0, 6424+books(%rip)
	movl	$0, 6428+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6432+books(%rip)
	movl	$0, 6440+books(%rip)
	movl	$0, 6444+books(%rip)
	movl	$0, 6448+books(%rip)
	movl	$0, 6452+books(%rip)
	movl	$0, 6456+books(%rip)
	movl	$0, 6460+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6464+books(%rip)
	movl	$0, 6472+books(%rip)
	movl	$0, 6476+books(%rip)
	movl	$0, 6480+books(%rip)
	movl	$0, 6484+books(%rip)
	movl	$0, 6488+books(%rip)
	movl	$0, 6492+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6496+books(%rip)
	movl	$0, 6504+books(%rip)
	movl	$0, 6508+books(%rip)
	movl	$0, 6512+books(%rip)
	movl	$0, 6516+books(%rip)
	movl	$0, 6520+books(%rip)
	movl	$0, 6524+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6528+books(%rip)
	movl	$0, 6536+books(%rip)
	movl	$0, 6540+books(%rip)
	movl	$0, 6544+books(%rip)
	movl	$0, 6548+books(%rip)
	movl	$0, 6552+books(%rip)
	movl	$0, 6556+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6560+books(%rip)
	movl	$0, 6568+books(%rip)
	movl	$0, 6572+books(%rip)
	movl	$0, 6576+books(%rip)
	movl	$0, 6580+books(%rip)
	movl	$0, 6584+books(%rip)
	movl	$0, 6588+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6592+books(%rip)
	movl	$0, 6600+books(%rip)
	movl	$0, 6604+books(%rip)
	movl	$0, 6608+books(%rip)
	movl	$0, 6612+books(%rip)
	movl	$0, 6616+books(%rip)
	movl	$0, 6620+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6624+books(%rip)
	movl	$0, 6632+books(%rip)
	movl	$0, 6636+books(%rip)
	movl	$0, 6640+books(%rip)
	movl	$0, 6644+books(%rip)
	movl	$0, 6648+books(%rip)
	movl	$0, 6652+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6656+books(%rip)
	movl	$0, 6664+books(%rip)
	movl	$0, 6668+books(%rip)
	movl	$0, 6672+books(%rip)
	movl	$0, 6676+books(%rip)
	movl	$0, 6680+books(%rip)
	movl	$0, 6684+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6688+books(%rip)
	movl	$0, 6696+books(%rip)
	movl	$0, 6700+books(%rip)
	movl	$0, 6704+books(%rip)
	movl	$0, 6708+books(%rip)
	movl	$0, 6712+books(%rip)
	movl	$0, 6716+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6720+books(%rip)
	movl	$0, 6728+books(%rip)
	movl	$0, 6732+books(%rip)
	movl	$0, 6736+books(%rip)
	movl	$0, 6740+books(%rip)
	movl	$0, 6744+books(%rip)
	movl	$0, 6748+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6752+books(%rip)
	movl	$0, 6760+books(%rip)
	movl	$0, 6764+books(%rip)
	movl	$0, 6768+books(%rip)
	movl	$0, 6772+books(%rip)
	movl	$0, 6776+books(%rip)
	movl	$0, 6780+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6784+books(%rip)
	movl	$0, 6792+books(%rip)
	movl	$0, 6796+books(%rip)
	movl	$0, 6800+books(%rip)
	movl	$0, 6804+books(%rip)
	movl	$0, 6808+books(%rip)
	movl	$0, 6812+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6816+books(%rip)
	movl	$0, 6824+books(%rip)
	movl	$0, 6828+books(%rip)
	movl	$0, 6832+books(%rip)
	movl	$0, 6836+books(%rip)
	movl	$0, 6840+books(%rip)
	movl	$0, 6844+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6848+books(%rip)
	movl	$0, 6856+books(%rip)
	movl	$0, 6860+books(%rip)
	movl	$0, 6864+books(%rip)
	movl	$0, 6868+books(%rip)
	movl	$0, 6872+books(%rip)
	movl	$0, 6876+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6880+books(%rip)
	movl	$0, 6888+books(%rip)
	movl	$0, 6892+books(%rip)
	movl	$0, 6896+books(%rip)
	movl	$0, 6900+books(%rip)
	movl	$0, 6904+books(%rip)
	movl	$0, 6908+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6912+books(%rip)
	movl	$0, 6920+books(%rip)
	movl	$0, 6924+books(%rip)
	movl	$0, 6928+books(%rip)
	movl	$0, 6932+books(%rip)
	movl	$0, 6936+books(%rip)
	movl	$0, 6940+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6944+books(%rip)
	movl	$0, 6952+books(%rip)
	movl	$0, 6956+books(%rip)
	movl	$0, 6960+books(%rip)
	movl	$0, 6964+books(%rip)
	movl	$0, 6968+books(%rip)
	movl	$0, 6972+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 6976+books(%rip)
	movl	$0, 6984+books(%rip)
	movl	$0, 6988+books(%rip)
	movl	$0, 6992+books(%rip)
	movl	$0, 6996+books(%rip)
	movl	$0, 7000+books(%rip)
	movl	$0, 7004+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7008+books(%rip)
	movl	$0, 7016+books(%rip)
	movl	$0, 7020+books(%rip)
	movl	$0, 7024+books(%rip)
	movl	$0, 7028+books(%rip)
	movl	$0, 7032+books(%rip)
	movl	$0, 7036+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7040+books(%rip)
	movl	$0, 7048+books(%rip)
	movl	$0, 7052+books(%rip)
	movl	$0, 7056+books(%rip)
	movl	$0, 7060+books(%rip)
	movl	$0, 7064+books(%rip)
	movl	$0, 7068+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7072+books(%rip)
	movl	$0, 7080+books(%rip)
	movl	$0, 7084+books(%rip)
	movl	$0, 7088+books(%rip)
	movl	$0, 7092+books(%rip)
	movl	$0, 7096+books(%rip)
	movl	$0, 7100+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7104+books(%rip)
	movl	$0, 7112+books(%rip)
	movl	$0, 7116+books(%rip)
	movl	$0, 7120+books(%rip)
	movl	$0, 7124+books(%rip)
	movl	$0, 7128+books(%rip)
	movl	$0, 7132+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7136+books(%rip)
	movl	$0, 7144+books(%rip)
	movl	$0, 7148+books(%rip)
	movl	$0, 7152+books(%rip)
	movl	$0, 7156+books(%rip)
	movl	$0, 7160+books(%rip)
	movl	$0, 7164+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7168+books(%rip)
	movl	$0, 7176+books(%rip)
	movl	$0, 7180+books(%rip)
	movl	$0, 7184+books(%rip)
	movl	$0, 7188+books(%rip)
	movl	$0, 7192+books(%rip)
	movl	$0, 7196+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7200+books(%rip)
	movl	$0, 7208+books(%rip)
	movl	$0, 7212+books(%rip)
	movl	$0, 7216+books(%rip)
	movl	$0, 7220+books(%rip)
	movl	$0, 7224+books(%rip)
	movl	$0, 7228+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7232+books(%rip)
	movl	$0, 7240+books(%rip)
	movl	$0, 7244+books(%rip)
	movl	$0, 7248+books(%rip)
	movl	$0, 7252+books(%rip)
	movl	$0, 7256+books(%rip)
	movl	$0, 7260+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7264+books(%rip)
	movl	$0, 7272+books(%rip)
	movl	$0, 7276+books(%rip)
	movl	$0, 7280+books(%rip)
	movl	$0, 7284+books(%rip)
	movl	$0, 7288+books(%rip)
	movl	$0, 7292+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7296+books(%rip)
	movl	$0, 7304+books(%rip)
	movl	$0, 7308+books(%rip)
	movl	$0, 7312+books(%rip)
	movl	$0, 7316+books(%rip)
	movl	$0, 7320+books(%rip)
	movl	$0, 7324+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7328+books(%rip)
	movl	$0, 7336+books(%rip)
	movl	$0, 7340+books(%rip)
	movl	$0, 7344+books(%rip)
	movl	$0, 7348+books(%rip)
	movl	$0, 7352+books(%rip)
	movl	$0, 7356+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7360+books(%rip)
	movl	$0, 7368+books(%rip)
	movl	$0, 7372+books(%rip)
	movl	$0, 7376+books(%rip)
	movl	$0, 7380+books(%rip)
	movl	$0, 7384+books(%rip)
	movl	$0, 7388+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7392+books(%rip)
	movl	$0, 7400+books(%rip)
	movl	$0, 7404+books(%rip)
	movl	$0, 7408+books(%rip)
	movl	$0, 7412+books(%rip)
	movl	$0, 7416+books(%rip)
	movl	$0, 7420+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7424+books(%rip)
	movl	$0, 7432+books(%rip)
	movl	$0, 7436+books(%rip)
	movl	$0, 7440+books(%rip)
	movl	$0, 7444+books(%rip)
	movl	$0, 7448+books(%rip)
	movl	$0, 7452+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7456+books(%rip)
	movl	$0, 7464+books(%rip)
	movl	$0, 7468+books(%rip)
	movl	$0, 7472+books(%rip)
	movl	$0, 7476+books(%rip)
	movl	$0, 7480+books(%rip)
	movl	$0, 7484+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7488+books(%rip)
	movl	$0, 7496+books(%rip)
	movl	$0, 7500+books(%rip)
	movl	$0, 7504+books(%rip)
	movl	$0, 7508+books(%rip)
	movl	$0, 7512+books(%rip)
	movl	$0, 7516+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7520+books(%rip)
	movl	$0, 7528+books(%rip)
	movl	$0, 7532+books(%rip)
	movl	$0, 7536+books(%rip)
	movl	$0, 7540+books(%rip)
	movl	$0, 7544+books(%rip)
	movl	$0, 7548+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7552+books(%rip)
	movl	$0, 7560+books(%rip)
	movl	$0, 7564+books(%rip)
	movl	$0, 7568+books(%rip)
	movl	$0, 7572+books(%rip)
	movl	$0, 7576+books(%rip)
	movl	$0, 7580+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7584+books(%rip)
	movl	$0, 7592+books(%rip)
	movl	$0, 7596+books(%rip)
	movl	$0, 7600+books(%rip)
	movl	$0, 7604+books(%rip)
	movl	$0, 7608+books(%rip)
	movl	$0, 7612+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7616+books(%rip)
	movl	$0, 7624+books(%rip)
	movl	$0, 7628+books(%rip)
	movl	$0, 7632+books(%rip)
	movl	$0, 7636+books(%rip)
	movl	$0, 7640+books(%rip)
	movl	$0, 7644+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7648+books(%rip)
	movl	$0, 7656+books(%rip)
	movl	$0, 7660+books(%rip)
	movl	$0, 7664+books(%rip)
	movl	$0, 7668+books(%rip)
	movl	$0, 7672+books(%rip)
	movl	$0, 7676+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7680+books(%rip)
	movl	$0, 7688+books(%rip)
	movl	$0, 7692+books(%rip)
	movl	$0, 7696+books(%rip)
	movl	$0, 7700+books(%rip)
	movl	$0, 7704+books(%rip)
	movl	$0, 7708+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7712+books(%rip)
	movl	$0, 7720+books(%rip)
	movl	$0, 7724+books(%rip)
	movl	$0, 7728+books(%rip)
	movl	$0, 7732+books(%rip)
	movl	$0, 7736+books(%rip)
	movl	$0, 7740+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7744+books(%rip)
	movl	$0, 7752+books(%rip)
	movl	$0, 7756+books(%rip)
	movl	$0, 7760+books(%rip)
	movl	$0, 7764+books(%rip)
	movl	$0, 7768+books(%rip)
	movl	$0, 7772+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7776+books(%rip)
	movl	$0, 7784+books(%rip)
	movl	$0, 7788+books(%rip)
	movl	$0, 7792+books(%rip)
	movl	$0, 7796+books(%rip)
	movl	$0, 7800+books(%rip)
	movl	$0, 7804+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7808+books(%rip)
	movl	$0, 7816+books(%rip)
	movl	$0, 7820+books(%rip)
	movl	$0, 7824+books(%rip)
	movl	$0, 7828+books(%rip)
	movl	$0, 7832+books(%rip)
	movl	$0, 7836+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7840+books(%rip)
	movl	$0, 7848+books(%rip)
	movl	$0, 7852+books(%rip)
	movl	$0, 7856+books(%rip)
	movl	$0, 7860+books(%rip)
	movl	$0, 7864+books(%rip)
	movl	$0, 7868+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7872+books(%rip)
	movl	$0, 7880+books(%rip)
	movl	$0, 7884+books(%rip)
	movl	$0, 7888+books(%rip)
	movl	$0, 7892+books(%rip)
	movl	$0, 7896+books(%rip)
	movl	$0, 7900+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7904+books(%rip)
	movl	$0, 7912+books(%rip)
	movl	$0, 7916+books(%rip)
	movl	$0, 7920+books(%rip)
	movl	$0, 7924+books(%rip)
	movl	$0, 7928+books(%rip)
	movl	$0, 7932+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7936+books(%rip)
	movl	$0, 7944+books(%rip)
	movl	$0, 7948+books(%rip)
	movl	$0, 7952+books(%rip)
	movl	$0, 7956+books(%rip)
	movl	$0, 7960+books(%rip)
	movl	$0, 7964+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 7968+books(%rip)
	movl	$0, 7976+books(%rip)
	movl	$0, 7980+books(%rip)
	movl	$0, 7984+books(%rip)
	movl	$0, 7988+books(%rip)
	movl	$0, 7992+books(%rip)
	movl	$0, 7996+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8000+books(%rip)
	movl	$0, 8008+books(%rip)
	movl	$0, 8012+books(%rip)
	movl	$0, 8016+books(%rip)
	movl	$0, 8020+books(%rip)
	movl	$0, 8024+books(%rip)
	movl	$0, 8028+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8032+books(%rip)
	movl	$0, 8040+books(%rip)
	movl	$0, 8044+books(%rip)
	movl	$0, 8048+books(%rip)
	movl	$0, 8052+books(%rip)
	movl	$0, 8056+books(%rip)
	movl	$0, 8060+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8064+books(%rip)
	movl	$0, 8072+books(%rip)
	movl	$0, 8076+books(%rip)
	movl	$0, 8080+books(%rip)
	movl	$0, 8084+books(%rip)
	movl	$0, 8088+books(%rip)
	movl	$0, 8092+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8096+books(%rip)
	movl	$0, 8104+books(%rip)
	movl	$0, 8108+books(%rip)
	movl	$0, 8112+books(%rip)
	movl	$0, 8116+books(%rip)
	movl	$0, 8120+books(%rip)
	movl	$0, 8124+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8128+books(%rip)
	movl	$0, 8136+books(%rip)
	movl	$0, 8140+books(%rip)
	movl	$0, 8144+books(%rip)
	movl	$0, 8148+books(%rip)
	movl	$0, 8152+books(%rip)
	movl	$0, 8156+books(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8160+books(%rip)
	movl	$0, 8168+books(%rip)
	movl	$0, 8172+books(%rip)
	movl	$0, 8176+books(%rip)
	movl	$0, 8180+books(%rip)
	movl	$0, 8184+books(%rip)
	movl	$0, 8188+books(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_dk3g_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_dk3g_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_dk3g_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 1948 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dk3g--0
# 0 "" 2
#NO_APP
	movl	-580(%rbp), %eax
	movl	%eax, _TIG_IZ_dk3g_argc(%rip)
	movq	-592(%rbp), %rax
	movq	%rax, _TIG_IZ_dk3g_argv(%rip)
	movq	-600(%rbp), %rax
	movq	%rax, _TIG_IZ_dk3g_envp(%rip)
	nop
	movq	$1, -568(%rbp)
.L18:
	cmpq	$5, -568(%rbp)
	ja	.L21
	movq	-568(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L21-.L9
	.long	.L13-.L9
	.long	.L12-.L9
	.long	.L11-.L9
	.long	.L10-.L9
	.long	.L8-.L9
	.text
.L10:
	movl	$1, %eax
	jmp	.L19
.L13:
	cmpl	$3, -580(%rbp)
	je	.L15
	movq	$5, -568(%rbp)
	jmp	.L17
.L15:
	movq	$2, -568(%rbp)
	jmp	.L17
.L11:
	movl	$0, %eax
	jmp	.L19
.L8:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -568(%rbp)
	jmp	.L17
.L12:
	movq	-592(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -560(%rbp)
	movq	-592(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -552(%rbp)
	movq	-552(%rbp), %rdx
	movq	-560(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movabsq	$24560144848079140, %rax
	movq	%rax, -544(%rbp)
	movb	$0, -536(%rbp)
	movsd	.LC3(%rip), %xmm0
	movsd	%xmm0, -528(%rbp)
	movl	$0, -520(%rbp)
	movl	$0, -516(%rbp)
	movsd	.LC4(%rip), %xmm0
	movsd	%xmm0, -512(%rbp)
	movsd	.LC5(%rip), %xmm0
	movsd	%xmm0, -504(%rbp)
	movsd	.LC5(%rip), %xmm0
	movsd	%xmm0, -496(%rbp)
	movl	$0, -488(%rbp)
	movl	$0, -484(%rbp)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -480(%rbp)
	leaq	-544(%rbp), %rax
	addq	$72, %rax
	movabsq	$4049921550391526465, %rcx
	movq	%rcx, (%rax)
	movw	$56, 8(%rax)
	leaq	-544(%rbp), %rax
	addq	$108, %rax
	movabsq	$32199629384934724, %rcx
	movq	%rcx, (%rax)
	movb	$0, -392(%rbp)
	movl	$0, -388(%rbp)
	movl	$0, -384(%rbp)
	movl	$0, -380(%rbp)
	movl	$0, -572(%rbp)
	movq	-552(%rbp), %r9
	movq	-560(%rbp), %r8
	subq	$528, %rsp
	movq	%rsp, %rax
	movq	%rax, %rdi
	leaq	-544(%rbp), %rax
	movl	$66, %edx
	movq	%rax, %rsi
	movq	%rdx, %rcx
	rep movsq
	movq	%r9, %rsi
	movq	%r8, %rdi
	call	readBookFile
	addq	$528, %rsp
	movl	%eax, -572(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -568(%rbp)
	jmp	.L17
.L21:
	nop
.L17:
	jmp	.L18
.L19:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	call	__stack_chk_fail@PLT
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
.LC7:
	.string	"w"
.LC8:
	.string	"File could not be opened."
.LC9:
	.string	"No memory"
.LC10:
	.string	";"
.LC12:
	.string	"r"
	.text
	.globl	readBookFile
	.type	readBookFile, @function
readBookFile:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movq	%rdi, -216(%rbp)
	movq	%rsi, -224(%rbp)
	movq	$13, -152(%rbp)
.L56:
	cmpq	$22, -152(%rbp)
	ja	.L57
	movq	-152(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L57-.L25
	.long	.L43-.L25
	.long	.L42-.L25
	.long	.L57-.L25
	.long	.L41-.L25
	.long	.L40-.L25
	.long	.L39-.L25
	.long	.L38-.L25
	.long	.L37-.L25
	.long	.L36-.L25
	.long	.L35-.L25
	.long	.L34-.L25
	.long	.L57-.L25
	.long	.L33-.L25
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L30-.L25
	.long	.L29-.L25
	.long	.L57-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L24-.L25
	.text
.L41:
	movq	-192(%rbp), %rdx
	movq	-184(%rbp), %rax
	movl	$255, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -160(%rbp)
	movq	$6, -152(%rbp)
	jmp	.L44
.L32:
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -176(%rbp)
	movq	$19, -152(%rbp)
	jmp	.L44
.L31:
	movq	-224(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -144(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -200(%rbp)
	movq	$0, -136(%rbp)
	movq	$512, -136(%rbp)
	movq	-200(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$8, %esi
	leaq	16(%rbp), %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	24(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	32(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	40(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	44(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	48(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	56(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	64(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	72(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	76(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	80(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$8, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	88(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$32, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	120(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	124(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$32, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	156(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$12, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	168(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	172(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	176(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	180(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-200(%rbp), %rdx
	leaq	184(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$354, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$256, %edi
	call	malloc@PLT
	movq	%rax, -128(%rbp)
	movq	-128(%rbp), %rax
	movq	%rax, -184(%rbp)
	movq	$10, -152(%rbp)
	jmp	.L44
.L37:
	movq	-168(%rbp), %rax
	leaq	-1(%rax), %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L45
	movq	$1, -152(%rbp)
	jmp	.L44
.L45:
	movq	$5, -152(%rbp)
	jmp	.L44
.L43:
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	leaq	-1(%rax), %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$5, -152(%rbp)
	jmp	.L44
.L30:
	movl	$0, -204(%rbp)
	movq	$4, -152(%rbp)
	jmp	.L44
.L26:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -152(%rbp)
	jmp	.L44
.L34:
	movl	$0, %eax
	jmp	.L47
.L36:
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$11, -152(%rbp)
	jmp	.L44
.L33:
	movq	$15, -152(%rbp)
	jmp	.L44
.L28:
	cmpq	$0, -176(%rbp)
	je	.L48
	movq	$2, -152(%rbp)
	jmp	.L44
.L48:
	movq	$5, -152(%rbp)
	jmp	.L44
.L29:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -152(%rbp)
	jmp	.L44
.L39:
	cmpq	$0, -160(%rbp)
	je	.L50
	movq	$14, -152(%rbp)
	jmp	.L44
.L50:
	movq	$9, -152(%rbp)
	jmp	.L44
.L24:
	movl	$1, %eax
	jmp	.L47
.L40:
	movq	-184(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -112(%rbp)
	movq	-112(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	books(%rip), %rdx
	movq	%rax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -96(%rbp)
	movsd	-96(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	8+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -80(%rbp)
	movsd	-80(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	12+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -64(%rbp)
	movsd	-64(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	16+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -48(%rbp)
	movsd	-48(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	20+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movsd	-32(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	24+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movsd	-16(%rbp), %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	-204(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	leaq	28+books(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movl	-204(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	leaq	books(%rip), %rax
	leaq	(%rdx,%rax), %rdi
	movq	-200(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$32, %esi
	call	fwrite@PLT
	addl	$1, -204(%rbp)
	movq	$4, -152(%rbp)
	jmp	.L44
.L35:
	cmpq	$0, -184(%rbp)
	jne	.L52
	movq	$17, -152(%rbp)
	jmp	.L44
.L52:
	movq	$20, -152(%rbp)
	jmp	.L44
.L38:
	cmpq	$0, -192(%rbp)
	jne	.L54
	movq	$21, -152(%rbp)
	jmp	.L44
.L54:
	movq	$16, -152(%rbp)
	jmp	.L44
.L42:
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -168(%rbp)
	movq	$8, -152(%rbp)
	jmp	.L44
.L27:
	movq	-216(%rbp), %rax
	leaq	.LC12(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -192(%rbp)
	movq	$7, -152(%rbp)
	jmp	.L44
.L57:
	nop
.L44:
	jmp	.L56
.L47:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	readBookFile, .-readBookFile
	.section	.rodata
	.align 8
.LC3:
	.long	1717986918
	.long	1075930726
	.align 8
.LC4:
	.long	0
	.long	1079574528
	.align 8
.LC5:
	.long	-500134854
	.long	1044740494
	.align 8
.LC11:
	.long	0
	.long	1100470148
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
