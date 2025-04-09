	.file	"StomperHK_trabalho-final-lp_main_flatten.c"
	.text
	.globl	_TIG_IZ_j7Ss_argv
	.bss
	.align 8
	.type	_TIG_IZ_j7Ss_argv, @object
	.size	_TIG_IZ_j7Ss_argv, 8
_TIG_IZ_j7Ss_argv:
	.zero	8
	.globl	_TIG_IZ_j7Ss_argc
	.align 4
	.type	_TIG_IZ_j7Ss_argc, @object
	.size	_TIG_IZ_j7Ss_argc, 4
_TIG_IZ_j7Ss_argc:
	.zero	4
	.globl	_TIG_IZ_j7Ss_envp
	.align 8
	.type	_TIG_IZ_j7Ss_envp, @object
	.size	_TIG_IZ_j7Ss_envp, 8
_TIG_IZ_j7Ss_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"pagadores.txt"
.LC2:
	.string	"contas.txt"
.LC3:
	.string	"%d|%49[^|]|%19[^|]|%14[^\n]\n"
	.align 8
.LC4:
	.string	"Erro ao abrir arquivo de contas"
.LC5:
	.string	"Pagadores carregados: %d\n"
.LC6:
	.string	"Contas carregadas: %d\n"
.LC7:
	.string	"%d|%f|%19[^|]|%9[^|]|%d\n"
	.align 8
.LC8:
	.string	"Erro ao abrir arquivo de pagadores"
	.text
	.globl	carregar_dados
	.type	carregar_dados, @function
carregar_dados:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movq	%rcx, -80(%rbp)
	movq	%r8, -88(%rbp)
	movl	%r9d, -72(%rbp)
	movq	$15, -24(%rbp)
.L38:
	cmpq	$26, -24(%rbp)
	ja	.L39
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L39-.L4
	.long	.L39-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L39-.L4
	.long	.L39-.L4
	.long	.L14-.L4
	.long	.L39-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L39-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L40-.L4
	.long	.L39-.L4
	.long	.L3-.L4
	.text
.L10:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$10, -24(%rbp)
	jmp	.L24
.L19:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-64(%rbp), %rax
	movl	%edx, (%rax)
	movq	$19, -24(%rbp)
	jmp	.L24
.L13:
	movq	$23, -24(%rbp)
	jmp	.L24
.L17:
	movq	-88(%rbp), %rax
	movl	$0, (%rax)
	movq	$17, -24(%rbp)
	jmp	.L24
.L22:
	cmpl	$5, -44(%rbp)
	jne	.L25
	movq	$13, -24(%rbp)
	jmp	.L24
.L25:
	movq	$9, -24(%rbp)
	jmp	.L24
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$26, -24(%rbp)
	jmp	.L24
.L20:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	74(%rax), %rsi
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	54(%rax), %rcx
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdx
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cltq
	imulq	$92, %rax, %rdi
	movq	-56(%rbp), %rax
	addq	%rdi, %rax
	movq	%rax, %rdi
	movq	-40(%rbp), %rax
	movq	%rsi, %r9
	movq	%rcx, %r8
	movq	%rdx, %rcx
	movq	%rdi, %rdx
	leaq	.LC3(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -48(%rbp)
	movq	$20, -24(%rbp)
	jmp	.L24
.L12:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$22, -24(%rbp)
	jmp	.L24
.L3:
	cmpq	$0, -40(%rbp)
	jne	.L28
	movq	$7, -24(%rbp)
	jmp	.L24
.L28:
	movq	$2, -24(%rbp)
	jmp	.L24
.L16:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$22, -24(%rbp)
	jmp	.L24
.L14:
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-88(%rbp), %rax
	movl	%edx, (%rax)
	movq	$17, -24(%rbp)
	jmp	.L24
.L9:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -68(%rbp)
	jle	.L30
	movq	$3, -24(%rbp)
	jmp	.L24
.L30:
	movq	$18, -24(%rbp)
	jmp	.L24
.L11:
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -72(%rbp)
	jle	.L32
	movq	$0, -24(%rbp)
	jmp	.L24
.L32:
	movq	$9, -24(%rbp)
	jmp	.L24
.L7:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -24(%rbp)
	jmp	.L24
.L15:
	cmpq	$0, -32(%rbp)
	jne	.L34
	movq	$16, -24(%rbp)
	jmp	.L24
.L34:
	movq	$8, -24(%rbp)
	jmp	.L24
.L23:
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	leaq	40(%rax), %rsi
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	leaq	28(%rax), %r8
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rcx
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	subq	$8, %rsp
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	leaq	.LC7(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	addq	$16, %rsp
	movl	%eax, -44(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L24
.L18:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$10, -24(%rbp)
	jmp	.L24
.L21:
	movq	-64(%rbp), %rax
	movl	$0, (%rax)
	movq	$19, -24(%rbp)
	jmp	.L24
.L8:
	cmpl	$4, -48(%rbp)
	jne	.L36
	movq	$4, -24(%rbp)
	jmp	.L24
.L36:
	movq	$18, -24(%rbp)
	jmp	.L24
.L39:
	nop
.L24:
	jmp	.L38
.L40:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	carregar_dados, .-carregar_dados
	.section	.rodata
.LC9:
	.string	"Conta cadastrada com sucesso!"
	.align 8
.LC10:
	.string	"Limite m\303\241ximo de contas atingido!"
.LC11:
	.string	"Digite o valor da conta: "
.LC12:
	.string	"%f"
	.align 8
.LC13:
	.string	"Digite o status da conta (Pago/Pendente): "
.LC14:
	.string	"%s"
	.align 8
.LC15:
	.string	"Digite a data de vencimento (YYYY-MM-DD): "
	.align 8
.LC16:
	.string	"Digite o ID do pagador associado: "
.LC17:
	.string	"%d"
	.align 8
.LC18:
	.string	"Erro: J\303\241 existe uma conta com esse ID."
.LC19:
	.string	"Digite o ID da conta: "
	.align 8
.LC20:
	.string	"Erro: Pagador com ID %d n\303\243o encontrado.\n"
	.text
	.globl	cadastrar_conta
	.type	cadastrar_conta, @function
cadastrar_conta:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$120, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movl	%ecx, -124(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$12, -88(%rbp)
.L65:
	cmpq	$15, -88(%rbp)
	ja	.L68
	movq	-88(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L44(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L44(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L44:
	.long	.L56-.L44
	.long	.L55-.L44
	.long	.L54-.L44
	.long	.L69-.L44
	.long	.L68-.L44
	.long	.L69-.L44
	.long	.L51-.L44
	.long	.L68-.L44
	.long	.L50-.L44
	.long	.L69-.L44
	.long	.L48-.L44
	.long	.L68-.L44
	.long	.L47-.L44
	.long	.L69-.L44
	.long	.L45-.L44
	.long	.L43-.L44
	.text
.L45:
	movq	-112(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-48(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movl	-40(%rbp), %edx
	movl	%edx, 40(%rax)
	movq	-112(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-112(%rbp), %rax
	movl	%edx, (%rax)
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -88(%rbp)
	jmp	.L57
.L43:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -88(%rbp)
	jmp	.L57
.L47:
	movq	-112(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$99, %eax
	jle	.L58
	movq	$15, -88(%rbp)
	jmp	.L57
.L58:
	movq	$10, -88(%rbp)
	jmp	.L57
.L50:
	cmpl	$0, -96(%rbp)
	je	.L60
	movq	$6, -88(%rbp)
	jmp	.L57
.L60:
	movq	$1, -88(%rbp)
	jmp	.L57
.L55:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	addq	$28, %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	addq	$40, %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-40(%rbp), %edx
	movl	-124(%rbp), %ecx
	movq	-120(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	verificar_pagador_existente
	movl	%eax, -92(%rbp)
	movq	$2, -88(%rbp)
	jmp	.L57
.L51:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -88(%rbp)
	jmp	.L57
.L48:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-80(%rbp), %edx
	movq	-112(%rbp), %rax
	movl	(%rax), %ecx
	movq	-104(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	verificar_conta_existente
	movl	%eax, -96(%rbp)
	movq	$8, -88(%rbp)
	jmp	.L57
.L56:
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -88(%rbp)
	jmp	.L57
.L54:
	cmpl	$0, -92(%rbp)
	je	.L63
	movq	$14, -88(%rbp)
	jmp	.L57
.L63:
	movq	$0, -88(%rbp)
	jmp	.L57
.L68:
	nop
.L57:
	jmp	.L65
.L69:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L67
	call	__stack_chk_fail@PLT
.L67:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	cadastrar_conta, .-cadastrar_conta
	.section	.rodata
	.align 8
.LC21:
	.string	"ID: %d|Nome:%s|CPF/CNJP: %s|TEL: %s\n"
	.align 8
.LC22:
	.string	"Erro ao abrir os arquivos para salvar os dados!"
.LC23:
	.string	"Dados salvos com sucesso!"
	.align 8
.LC24:
	.string	"ID: %d|VALOR: %.2f| STATUS: %s| VALIDADE: %s| ID (Padagor):%d\n"
.LC25:
	.string	"w"
	.text
	.globl	salvar_dados
	.type	salvar_dados, @function
salvar_dados:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movq	%rdx, -72(%rbp)
	movl	%ecx, -64(%rbp)
	movq	$14, -24(%rbp)
.L99:
	cmpq	$20, -24(%rbp)
	ja	.L100
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L73(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L73(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L73:
	.long	.L88-.L73
	.long	.L87-.L73
	.long	.L86-.L73
	.long	.L85-.L73
	.long	.L84-.L73
	.long	.L100-.L73
	.long	.L100-.L73
	.long	.L83-.L73
	.long	.L101-.L73
	.long	.L81-.L73
	.long	.L101-.L73
	.long	.L100-.L73
	.long	.L101-.L73
	.long	.L78-.L73
	.long	.L77-.L73
	.long	.L100-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L74-.L73
	.long	.L100-.L73
	.long	.L72-.L73
	.text
.L74:
	movl	$0, -44(%rbp)
	movq	$9, -24(%rbp)
	jmp	.L89
.L84:
	movl	-48(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	74(%rax), %rdi
	movl	-48(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	54(%rax), %rsi
	movl	-48(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rcx
	movl	-48(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-40(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	leaq	.LC21(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	addl	$1, -48(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L89
.L77:
	movq	$20, -24(%rbp)
	jmp	.L89
.L87:
	movl	-48(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L91
	movq	$4, -24(%rbp)
	jmp	.L89
.L91:
	movq	$18, -24(%rbp)
	jmp	.L89
.L85:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -24(%rbp)
	jmp	.L89
.L76:
	cmpq	$0, -40(%rbp)
	jne	.L93
	movq	$2, -24(%rbp)
	jmp	.L89
.L93:
	movq	$0, -24(%rbp)
	jmp	.L89
.L81:
	movl	-44(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L95
	movq	$7, -24(%rbp)
	jmp	.L89
.L95:
	movq	$13, -24(%rbp)
	jmp	.L89
.L78:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -24(%rbp)
	jmp	.L89
.L75:
	movl	$0, -48(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L89
.L88:
	cmpq	$0, -32(%rbp)
	jne	.L97
	movq	$3, -24(%rbp)
	jmp	.L89
.L97:
	movq	$17, -24(%rbp)
	jmp	.L89
.L83:
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	40(%rax), %edi
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	leaq	28(%rax), %r8
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movss	4(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rsi
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edi, %r9d
	movq	%rsi, %xmm0
	leaq	.LC24(%rip), %rsi
	movq	%rax, %rdi
	movl	$1, %eax
	call	fprintf@PLT
	addl	$1, -44(%rbp)
	movq	$9, -24(%rbp)
	jmp	.L89
.L86:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -24(%rbp)
	jmp	.L89
.L72:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	leaq	.LC25(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$16, -24(%rbp)
	jmp	.L89
.L100:
	nop
.L89:
	jmp	.L99
.L101:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	salvar_dados, .-salvar_dados
	.section	.rodata
.LC26:
	.string	"\nContas cadastradas:"
	.align 8
.LC27:
	.string	"ID Conta: %d | Valor: %.2f | Status: %s | Vencimento: %s | Pagador: %s (ID: %d)\n"
	.text
	.globl	listar_contas
	.type	listar_contas, @function
listar_contas:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -104(%rbp)
	movl	%esi, -108(%rbp)
	movq	%rdx, -120(%rbp)
	movl	%ecx, -112(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -72(%rbp)
.L128:
	cmpq	$22, -72(%rbp)
	ja	.L131
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L105(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L105(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L105:
	.long	.L131-.L105
	.long	.L117-.L105
	.long	.L116-.L105
	.long	.L115-.L105
	.long	.L114-.L105
	.long	.L131-.L105
	.long	.L113-.L105
	.long	.L131-.L105
	.long	.L131-.L105
	.long	.L112-.L105
	.long	.L131-.L105
	.long	.L111-.L105
	.long	.L131-.L105
	.long	.L131-.L105
	.long	.L131-.L105
	.long	.L110-.L105
	.long	.L132-.L105
	.long	.L131-.L105
	.long	.L108-.L105
	.long	.L131-.L105
	.long	.L107-.L105
	.long	.L106-.L105
	.long	.L104-.L105
	.text
.L108:
	addl	$1, -76(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L118
.L114:
	movb	$68, -64(%rbp)
	movb	$101, -63(%rbp)
	movb	$115, -62(%rbp)
	movb	$99, -61(%rbp)
	movb	$111, -60(%rbp)
	movb	$110, -59(%rbp)
	movb	$104, -58(%rbp)
	movb	$101, -57(%rbp)
	movb	$99, -56(%rbp)
	movb	$105, -55(%rbp)
	movb	$100, -54(%rbp)
	movb	$111, -53(%rbp)
	movb	$0, -52(%rbp)
	movl	$13, -80(%rbp)
	movq	$6, -72(%rbp)
	jmp	.L118
.L110:
	movl	-76(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$20, -72(%rbp)
	jmp	.L118
.L117:
	movl	-76(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L119
	movq	$2, -72(%rbp)
	jmp	.L118
.L119:
	movq	$20, -72(%rbp)
	jmp	.L118
.L115:
	movq	$21, -72(%rbp)
	jmp	.L118
.L106:
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -84(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L118
.L111:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L122
	movq	$4, -72(%rbp)
	jmp	.L118
.L122:
	movq	$16, -72(%rbp)
	jmp	.L118
.L112:
	movl	-80(%rbp), %eax
	movb	$0, -64(%rbp,%rax)
	addl	$1, -80(%rbp)
	movq	$6, -72(%rbp)
	jmp	.L118
.L113:
	cmpl	$49, -80(%rbp)
	jbe	.L124
	movq	$22, -72(%rbp)
	jmp	.L118
.L124:
	movq	$9, -72(%rbp)
	jmp	.L118
.L104:
	movl	$0, -76(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L118
.L116:
	movl	-76(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	40(%rax), %eax
	cmpl	%eax, %ecx
	jne	.L126
	movq	$15, -72(%rbp)
	jmp	.L118
.L126:
	movq	$18, -72(%rbp)
	jmp	.L118
.L107:
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	40(%rax), %r8d
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	leaq	28(%rax), %rcx
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movss	4(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rsi
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	leaq	-64(%rbp), %rdx
	movl	%r8d, %r9d
	movq	%rdx, %r8
	movq	%rdi, %rdx
	movq	%rsi, %xmm0
	movl	%eax, %esi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -84(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L118
.L131:
	nop
.L118:
	jmp	.L128
.L132:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L130
	call	__stack_chk_fail@PLT
.L130:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	listar_contas, .-listar_contas
	.section	.rodata
	.align 8
.LC28:
	.string	"Limite m\303\241ximo de pagadores atingido!"
.LC29:
	.string	"Digite o nome do pagador: "
.LC30:
	.string	" %[^\n]"
	.align 8
.LC31:
	.string	"Digite o CPF/CNPJ do pagador: "
	.align 8
.LC32:
	.string	"Digite o telefone do pagador: "
	.align 8
.LC33:
	.string	"Pagador cadastrado com sucesso!"
	.align 8
.LC34:
	.string	"Erro: J\303\241 existe um pagador com esse ID."
.LC35:
	.string	"Digite o ID do pagador: "
	.text
	.globl	cadastrar_pagador
	.type	cadastrar_pagador, @function
cadastrar_pagador:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$152, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$6, -136(%rbp)
.L151:
	cmpq	$10, -136(%rbp)
	ja	.L154
	movq	-136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L136(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L136(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L136:
	.long	.L155-.L136
	.long	.L155-.L136
	.long	.L142-.L136
	.long	.L141-.L136
	.long	.L140-.L136
	.long	.L154-.L136
	.long	.L139-.L136
	.long	.L138-.L136
	.long	.L154-.L136
	.long	.L155-.L136
	.long	.L135-.L136
	.text
.L140:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -136(%rbp)
	jmp	.L145
.L141:
	cmpl	$0, -140(%rbp)
	je	.L147
	movq	$7, -136(%rbp)
	jmp	.L145
.L147:
	movq	$10, -136(%rbp)
	jmp	.L145
.L139:
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$99, %eax
	jle	.L149
	movq	$4, -136(%rbp)
	jmp	.L145
.L149:
	movq	$2, -136(%rbp)
	jmp	.L145
.L135:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	addq	$54, %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	addq	$74, %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-96(%rbp), %rcx
	movq	-88(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-48(%rbp), %rdx
	movq	%rdx, 80(%rax)
	movl	-40(%rbp), %edx
	movl	%edx, 88(%rax)
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-160(%rbp), %rax
	movl	%edx, (%rax)
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -136(%rbp)
	jmp	.L145
.L138:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -136(%rbp)
	jmp	.L145
.L142:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-128(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-128(%rbp), %edx
	movq	-160(%rbp), %rax
	movl	(%rax), %ecx
	movq	-152(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	verificar_pagador_existente
	movl	%eax, -140(%rbp)
	movq	$3, -136(%rbp)
	jmp	.L145
.L154:
	nop
.L145:
	jmp	.L151
.L155:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L153
	call	__stack_chk_fail@PLT
.L153:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	cadastrar_pagador, .-cadastrar_pagador
	.section	.rodata
	.align 8
.LC36:
	.string	"\nGerenciamento de Contas e Pagadores"
.LC37:
	.string	"1. Cadastrar Pagador"
.LC38:
	.string	"2. Cadastrar Conta"
.LC39:
	.string	"3. Listar Pagadores"
.LC40:
	.string	"4. Listar Contas"
.LC41:
	.string	"5. Buscar Pagador por ID"
.LC42:
	.string	"6. Buscar Conta por ID"
.LC43:
	.string	"7. Sair"
.LC44:
	.string	"Escolha uma op\303\247\303\243o: "
	.text
	.globl	menu
	.type	menu, @function
menu:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L162:
	cmpq	$2, -8(%rbp)
	je	.L157
	cmpq	$2, -8(%rbp)
	ja	.L163
	cmpq	$0, -8(%rbp)
	je	.L164
	cmpq	$1, -8(%rbp)
	jne	.L163
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L160
.L157:
	movq	$1, -8(%rbp)
	jmp	.L160
.L163:
	nop
.L160:
	jmp	.L162
.L164:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	menu, .-menu
	.globl	verificar_pagador_existente
	.type	verificar_pagador_existente, @function
verificar_pagador_existente:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$5, -8(%rbp)
.L180:
	cmpq	$7, -8(%rbp)
	ja	.L181
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L168(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L168(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L168:
	.long	.L181-.L168
	.long	.L173-.L168
	.long	.L172-.L168
	.long	.L181-.L168
	.long	.L171-.L168
	.long	.L170-.L168
	.long	.L169-.L168
	.long	.L167-.L168
	.text
.L171:
	movl	$0, %eax
	jmp	.L174
.L173:
	movl	$1, %eax
	jmp	.L174
.L169:
	movl	-12(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L175
	movq	$1, -8(%rbp)
	jmp	.L177
.L175:
	movq	$2, -8(%rbp)
	jmp	.L177
.L170:
	movl	$0, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L177
.L167:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L178
	movq	$6, -8(%rbp)
	jmp	.L177
.L178:
	movq	$4, -8(%rbp)
	jmp	.L177
.L172:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L177
.L181:
	nop
.L177:
	jmp	.L180
.L174:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	verificar_pagador_existente, .-verificar_pagador_existente
	.section	.rodata
	.align 8
.LC45:
	.string	"Digite o ID da conta que deseja buscar: "
.LC46:
	.string	"\nConta encontrada!"
	.align 8
.LC47:
	.string	"\nErro: Nenhuma conta encontrada com o ID %d.\n"
	.text
	.globl	buscar_conta_por_id
	.type	buscar_conta_por_id, @function
buscar_conta_por_id:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -104(%rbp)
	movl	%esi, -108(%rbp)
	movq	%rdx, -120(%rbp)
	movl	%ecx, -112(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -72(%rbp)
.L214:
	cmpq	$26, -72(%rbp)
	ja	.L217
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L185(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L185(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L185:
	.long	.L217-.L185
	.long	.L218-.L185
	.long	.L200-.L185
	.long	.L217-.L185
	.long	.L199-.L185
	.long	.L217-.L185
	.long	.L198-.L185
	.long	.L217-.L185
	.long	.L197-.L185
	.long	.L217-.L185
	.long	.L217-.L185
	.long	.L196-.L185
	.long	.L195-.L185
	.long	.L194-.L185
	.long	.L217-.L185
	.long	.L193-.L185
	.long	.L217-.L185
	.long	.L217-.L185
	.long	.L192-.L185
	.long	.L218-.L185
	.long	.L190-.L185
	.long	.L189-.L185
	.long	.L188-.L185
	.long	.L187-.L185
	.long	.L217-.L185
	.long	.L186-.L185
	.long	.L184-.L185
	.text
.L192:
	movl	-76(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$11, -72(%rbp)
	jmp	.L202
.L186:
	movl	-76(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L203
	movq	$22, -72(%rbp)
	jmp	.L202
.L203:
	movq	$11, -72(%rbp)
	jmp	.L202
.L199:
	movq	$23, -72(%rbp)
	jmp	.L202
.L193:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L205
	movq	$20, -72(%rbp)
	jmp	.L202
.L205:
	movq	$13, -72(%rbp)
	jmp	.L202
.L195:
	addl	$1, -76(%rbp)
	movq	$25, -72(%rbp)
	jmp	.L202
.L197:
	addl	$1, -84(%rbp)
	movq	$15, -72(%rbp)
	jmp	.L202
.L187:
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-88(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -84(%rbp)
	movq	$15, -72(%rbp)
	jmp	.L202
.L189:
	movb	$68, -64(%rbp)
	movb	$101, -63(%rbp)
	movb	$115, -62(%rbp)
	movb	$99, -61(%rbp)
	movb	$111, -60(%rbp)
	movb	$110, -59(%rbp)
	movb	$104, -58(%rbp)
	movb	$101, -57(%rbp)
	movb	$99, -56(%rbp)
	movb	$105, -55(%rbp)
	movb	$100, -54(%rbp)
	movb	$111, -53(%rbp)
	movb	$0, -52(%rbp)
	movl	$13, -80(%rbp)
	movq	$2, -72(%rbp)
	jmp	.L202
.L184:
	movl	-80(%rbp), %eax
	movb	$0, -64(%rbp,%rax)
	addl	$1, -80(%rbp)
	movq	$2, -72(%rbp)
	jmp	.L202
.L196:
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	40(%rax), %r8d
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	leaq	28(%rax), %rcx
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movss	4(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rsi
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	leaq	-64(%rbp), %rdx
	movl	%r8d, %r9d
	movq	%rdx, %r8
	movq	%rdi, %rdx
	movq	%rsi, %xmm0
	movl	%eax, %esi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$19, -72(%rbp)
	jmp	.L202
.L194:
	movl	-88(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -72(%rbp)
	jmp	.L202
.L198:
	movl	$0, -76(%rbp)
	movq	$25, -72(%rbp)
	jmp	.L202
.L188:
	movl	-76(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	40(%rax), %eax
	cmpl	%eax, %ecx
	jne	.L208
	movq	$18, -72(%rbp)
	jmp	.L202
.L208:
	movq	$12, -72(%rbp)
	jmp	.L202
.L200:
	cmpl	$49, -80(%rbp)
	jbe	.L210
	movq	$6, -72(%rbp)
	jmp	.L202
.L210:
	movq	$26, -72(%rbp)
	jmp	.L202
.L190:
	movl	-84(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-88(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L212
	movq	$21, -72(%rbp)
	jmp	.L202
.L212:
	movq	$8, -72(%rbp)
	jmp	.L202
.L217:
	nop
.L202:
	jmp	.L214
.L218:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L216
	call	__stack_chk_fail@PLT
.L216:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	buscar_conta_por_id, .-buscar_conta_por_id
	.section	.rodata
	.align 8
.LC48:
	.string	"Digite o ID do pagador que deseja buscar: "
	.align 8
.LC49:
	.string	"\nErro: Nenhum pagador encontrado com o ID %d.\n"
.LC50:
	.string	"\nPagador encontrado!"
	.align 8
.LC51:
	.string	"ID: %d | Nome: %s | CPF/CNPJ: %s | Telefone: %s\n"
	.text
	.globl	buscar_pagador_por_id
	.type	buscar_pagador_por_id, @function
buscar_pagador_por_id:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$11, -16(%rbp)
.L237:
	cmpq	$11, -16(%rbp)
	ja	.L240
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L222(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L222(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L222:
	.long	.L241-.L222
	.long	.L240-.L222
	.long	.L229-.L222
	.long	.L228-.L222
	.long	.L227-.L222
	.long	.L240-.L222
	.long	.L226-.L222
	.long	.L225-.L222
	.long	.L240-.L222
	.long	.L241-.L222
	.long	.L223-.L222
	.long	.L221-.L222
	.text
.L227:
	addl	$1, -20(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L231
.L228:
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -20(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L231
.L221:
	movq	$3, -16(%rbp)
	jmp	.L231
.L226:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L233
	movq	$2, -16(%rbp)
	jmp	.L231
.L233:
	movq	$10, -16(%rbp)
	jmp	.L231
.L223:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L231
.L225:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-20(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	leaq	74(%rax), %rsi
	movl	-20(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	leaq	54(%rax), %rcx
	movl	-20(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdx
	movl	-20(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdi
	movq	-40(%rbp), %rax
	addq	%rdi, %rax
	movl	(%rax), %eax
	movq	%rsi, %r8
	movl	%eax, %esi
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L231
.L229:
	movl	-20(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L235
	movq	$7, -16(%rbp)
	jmp	.L231
.L235:
	movq	$4, -16(%rbp)
	jmp	.L231
.L240:
	nop
.L231:
	jmp	.L237
.L241:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L239
	call	__stack_chk_fail@PLT
.L239:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	buscar_pagador_por_id, .-buscar_pagador_por_id
	.section	.rodata
.LC52:
	.string	"\nPagadores cadastrados:"
	.text
	.globl	listar_pagadores
	.type	listar_pagadores, @function
listar_pagadores:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$2, -8(%rbp)
.L254:
	cmpq	$7, -8(%rbp)
	ja	.L255
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L245(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L245(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L245:
	.long	.L249-.L245
	.long	.L255-.L245
	.long	.L248-.L245
	.long	.L247-.L245
	.long	.L256-.L245
	.long	.L255-.L245
	.long	.L255-.L245
	.long	.L244-.L245
	.text
.L247:
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L251
.L249:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L252
	movq	$7, -8(%rbp)
	jmp	.L251
.L252:
	movq	$4, -8(%rbp)
	jmp	.L251
.L244:
	movl	-12(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	leaq	74(%rax), %rsi
	movl	-12(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	leaq	54(%rax), %rcx
	movl	-12(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	imulq	$92, %rax, %rdi
	movq	-24(%rbp), %rax
	addq	%rdi, %rax
	movl	(%rax), %eax
	movq	%rsi, %r8
	movl	%eax, %esi
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L251
.L248:
	movq	$3, -8(%rbp)
	jmp	.L251
.L255:
	nop
.L251:
	jmp	.L254
.L256:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	listar_pagadores, .-listar_pagadores
	.section	.rodata
	.align 8
.LC53:
	.string	"Op\303\247\303\243o inv\303\241lida! Tente novamente."
.LC54:
	.string	"Saindo do programa..."
	.text
	.globl	main
	.type	main, @function
main:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$1392, %rsp
	movl	%edi, -13652(%rbp)
	movq	%rsi, -13664(%rbp)
	movq	%rdx, -13672(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_j7Ss_envp(%rip)
	nop
.L258:
	movq	$0, _TIG_IZ_j7Ss_argv(%rip)
	nop
.L259:
	movl	$0, _TIG_IZ_j7Ss_argc(%rip)
	nop
	nop
.L260:
.L261:
#APP
# 148 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-j7Ss--0
# 0 "" 2
#NO_APP
	movl	-13652(%rbp), %eax
	movl	%eax, _TIG_IZ_j7Ss_argc(%rip)
	movq	-13664(%rbp), %rax
	movq	%rax, _TIG_IZ_j7Ss_argv(%rip)
	movq	-13672(%rbp), %rax
	movq	%rax, _TIG_IZ_j7Ss_envp(%rip)
	nop
	movq	$3, -13624(%rbp)
.L292:
	cmpq	$22, -13624(%rbp)
	ja	.L295
	movq	-13624(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L264(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L264(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L264:
	.long	.L277-.L264
	.long	.L295-.L264
	.long	.L276-.L264
	.long	.L275-.L264
	.long	.L295-.L264
	.long	.L274-.L264
	.long	.L273-.L264
	.long	.L272-.L264
	.long	.L271-.L264
	.long	.L270-.L264
	.long	.L295-.L264
	.long	.L269-.L264
	.long	.L295-.L264
	.long	.L268-.L264
	.long	.L295-.L264
	.long	.L267-.L264
	.long	.L295-.L264
	.long	.L295-.L264
	.long	.L266-.L264
	.long	.L295-.L264
	.long	.L295-.L264
	.long	.L265-.L264
	.long	.L263-.L264
	.text
.L266:
	movl	-13632(%rbp), %ecx
	leaq	-9216(%rbp), %rdx
	leaq	-13636(%rbp), %rsi
	leaq	-13616(%rbp), %rax
	movq	%rax, %rdi
	call	cadastrar_conta
	movq	$6, -13624(%rbp)
	jmp	.L278
.L267:
	movl	$0, -13636(%rbp)
	movl	$0, -13632(%rbp)
	leaq	-13636(%rbp), %rcx
	leaq	-13616(%rbp), %rdx
	leaq	-13632(%rbp), %rsi
	leaq	-9216(%rbp), %rax
	movl	$100, %r9d
	movq	%rcx, %r8
	movq	%rdx, %rcx
	movl	$100, %edx
	movq	%rax, %rdi
	call	carregar_dados
	movq	$13, -13624(%rbp)
	jmp	.L278
.L271:
	movl	-13632(%rbp), %edx
	leaq	-9216(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	listar_pagadores
	movq	$6, -13624(%rbp)
	jmp	.L278
.L275:
	movq	$15, -13624(%rbp)
	jmp	.L278
.L265:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L293
	jmp	.L294
.L269:
	movl	-13632(%rbp), %ecx
	movl	-13636(%rbp), %esi
	leaq	-9216(%rbp), %rdx
	leaq	-13616(%rbp), %rax
	movq	%rax, %rdi
	call	buscar_conta_por_id
	movq	$6, -13624(%rbp)
	jmp	.L278
.L270:
	leaq	.LC53(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -13624(%rbp)
	jmp	.L278
.L268:
	call	menu
	leaq	-13628(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -13624(%rbp)
	jmp	.L278
.L273:
	movl	-13628(%rbp), %eax
	cmpl	$7, %eax
	je	.L280
	movq	$13, -13624(%rbp)
	jmp	.L278
.L280:
	movq	$21, -13624(%rbp)
	jmp	.L278
.L263:
	leaq	-13632(%rbp), %rdx
	leaq	-9216(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cadastrar_pagador
	movq	$6, -13624(%rbp)
	jmp	.L278
.L274:
	movl	-13632(%rbp), %edx
	leaq	-9216(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	buscar_pagador_por_id
	movq	$6, -13624(%rbp)
	jmp	.L278
.L277:
	movl	-13632(%rbp), %ecx
	movl	-13636(%rbp), %esi
	leaq	-9216(%rbp), %rdx
	leaq	-13616(%rbp), %rax
	movq	%rax, %rdi
	call	listar_contas
	movq	$6, -13624(%rbp)
	jmp	.L278
.L272:
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-13636(%rbp), %ecx
	movl	-13632(%rbp), %esi
	leaq	-13616(%rbp), %rdx
	leaq	-9216(%rbp), %rax
	movq	%rax, %rdi
	call	salvar_dados
	movq	$6, -13624(%rbp)
	jmp	.L278
.L276:
	movl	-13628(%rbp), %eax
	cmpl	$7, %eax
	ja	.L282
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L284(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L284(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L284:
	.long	.L282-.L284
	.long	.L290-.L284
	.long	.L289-.L284
	.long	.L288-.L284
	.long	.L287-.L284
	.long	.L286-.L284
	.long	.L285-.L284
	.long	.L283-.L284
	.text
.L283:
	movq	$7, -13624(%rbp)
	jmp	.L291
.L285:
	movq	$11, -13624(%rbp)
	jmp	.L291
.L286:
	movq	$5, -13624(%rbp)
	jmp	.L291
.L287:
	movq	$0, -13624(%rbp)
	jmp	.L291
.L288:
	movq	$8, -13624(%rbp)
	jmp	.L291
.L289:
	movq	$18, -13624(%rbp)
	jmp	.L291
.L290:
	movq	$22, -13624(%rbp)
	jmp	.L291
.L282:
	movq	$9, -13624(%rbp)
	nop
.L291:
	jmp	.L278
.L295:
	nop
.L278:
	jmp	.L292
.L294:
	call	__stack_chk_fail@PLT
.L293:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	main, .-main
	.globl	verificar_conta_existente
	.type	verificar_conta_existente, @function
verificar_conta_existente:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$3, -8(%rbp)
.L311:
	cmpq	$6, -8(%rbp)
	ja	.L312
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L299(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L299(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L299:
	.long	.L304-.L299
	.long	.L303-.L299
	.long	.L312-.L299
	.long	.L302-.L299
	.long	.L301-.L299
	.long	.L300-.L299
	.long	.L298-.L299
	.text
.L301:
	movl	$1, %eax
	jmp	.L305
.L303:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L306
	movq	$5, -8(%rbp)
	jmp	.L308
.L306:
	movq	$6, -8(%rbp)
	jmp	.L308
.L302:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L308
.L298:
	movl	$0, %eax
	jmp	.L305
.L300:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L309
	movq	$4, -8(%rbp)
	jmp	.L308
.L309:
	movq	$0, -8(%rbp)
	jmp	.L308
.L304:
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L308
.L312:
	nop
.L308:
	jmp	.L311
.L305:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	verificar_conta_existente, .-verificar_conta_existente
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
