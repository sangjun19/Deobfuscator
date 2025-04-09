	.file	"murilocpontes_Trabalho-2-Lab-ICC-biblioteca_main_windows_flatten.c"
	.text
	.globl	_TIG_IZ_Jvlc_argc
	.bss
	.align 4
	.type	_TIG_IZ_Jvlc_argc, @object
	.size	_TIG_IZ_Jvlc_argc, 4
_TIG_IZ_Jvlc_argc:
	.zero	4
	.globl	_TIG_IZ_Jvlc_argv
	.align 8
	.type	_TIG_IZ_Jvlc_argv, @object
	.size	_TIG_IZ_Jvlc_argv, 8
_TIG_IZ_Jvlc_argv:
	.zero	8
	.globl	_TIG_IZ_Jvlc_envp
	.align 8
	.type	_TIG_IZ_Jvlc_envp, @object
	.size	_TIG_IZ_Jvlc_envp, 8
_TIG_IZ_Jvlc_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"ab"
	.align 8
.LC1:
	.string	"Erro, nao foi possivel abrir o arquivo"
	.text
	.globl	salva_cadastro
	.type	salva_cadastro, @function
salva_cadastro:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$2, -16(%rbp)
.L15:
	cmpq	$6, -16(%rbp)
	ja	.L16
	movq	-16(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L17-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$4, -16(%rbp)
	jmp	.L12
.L7:
	movq	-48(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L12
.L3:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L12
.L5:
	cmpq	$0, -24(%rbp)
	jne	.L13
	movq	$6, -16(%rbp)
	jmp	.L12
.L13:
	movq	$0, -16(%rbp)
	jmp	.L12
.L10:
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, %rcx
	movl	$50, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	52(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	56(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	60(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	64(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	68(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	72(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$40, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	112(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	116(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$20, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	136(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$40, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	176(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$30, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	-40(%rbp), %rax
	leaq	206(%rax), %rdi
	movq	-24(%rbp), %rax
	movq	%rax, %rcx
	movl	$20, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	$1, -16(%rbp)
	jmp	.L12
.L8:
	movq	$3, -16(%rbp)
	jmp	.L12
.L16:
	nop
.L12:
	jmp	.L15
.L17:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	salva_cadastro, .-salva_cadastro
	.section	.rodata
.LC2:
	.string	"Quantidade de livros: %d\n"
	.align 8
.LC3:
	.string	"Livro %d:Data de emprestimo: %d/%d/%d\n"
	.align 8
.LC4:
	.string	"Livro %d:Data de devolucao: %d/%d/%d\n"
.LC5:
	.string	"Livro %d: %sAutor %s\n"
.LC6:
	.string	"r+b"
.LC7:
	.string	"Erro ao abrir o arquivo"
	.text
	.globl	imprimir_emprestimo
	.type	imprimir_emprestimo, @function
imprimir_emprestimo:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$752, %rsp
	movq	%rdi, -744(%rbp)
	movq	%rsi, -752(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$21, -704(%rbp)
.L55:
	cmpq	$32, -704(%rbp)
	ja	.L58
	movq	-704(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L40-.L21
	.long	.L39-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L38-.L21
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L58-.L21
	.long	.L35-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L34-.L21
	.long	.L58-.L21
	.long	.L58-.L21
	.long	.L33-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L58-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L59-.L21
	.long	.L58-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L58-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L32:
	cmpq	$0, -712(%rbp)
	jne	.L41
	movq	$4, -704(%rbp)
	jmp	.L43
.L41:
	movq	$22, -704(%rbp)
	jmp	.L43
.L38:
	movq	-720(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$25, -704(%rbp)
	jmp	.L43
.L23:
	movl	-632(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -728(%rbp)
	movq	$28, -704(%rbp)
	jmp	.L43
.L34:
	movq	-720(%rbp), %rax
	movl	$1, %edx
	movl	$628, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-720(%rbp), %rdx
	leaq	-729(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -712(%rbp)
	movq	$18, -704(%rbp)
	jmp	.L43
.L22:
	cmpl	$0, -724(%rbp)
	jne	.L45
	movq	$1, -704(%rbp)
	jmp	.L43
.L45:
	movq	$14, -704(%rbp)
	jmp	.L43
.L35:
	cmpq	$0, -720(%rbp)
	jne	.L47
	movq	$27, -704(%rbp)
	jmp	.L43
.L47:
	movq	$23, -704(%rbp)
	jmp	.L43
.L39:
	movq	-720(%rbp), %rax
	leaq	-688(%rbp), %rdx
	leaq	52(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-720(%rbp), %rax
	leaq	-688(%rbp), %rdx
	leaq	56(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movl	$0, -728(%rbp)
	movq	$24, -704(%rbp)
	jmp	.L43
.L28:
	movq	-720(%rbp), %rdx
	leaq	-688(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movq	-744(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -724(%rbp)
	movq	$31, -704(%rbp)
	jmp	.L43
.L27:
	movl	-632(%rbp), %eax
	cmpl	%eax, -728(%rbp)
	jge	.L49
	movq	$5, -704(%rbp)
	jmp	.L43
.L49:
	movq	$30, -704(%rbp)
	jmp	.L43
.L30:
	movq	$17, -704(%rbp)
	jmp	.L43
.L31:
	movl	-728(%rbp), %eax
	cltq
	addq	$24, %rax
	movl	-684(%rbp,%rax,4), %edi
	movl	-728(%rbp), %eax
	cltq
	addq	$20, %rax
	movl	-688(%rbp,%rax,4), %edx
	movl	-728(%rbp), %eax
	cltq
	addq	$12, %rax
	movl	-676(%rbp,%rax,4), %eax
	movl	-728(%rbp), %ecx
	leal	1(%rcx), %esi
	movl	%edi, %r8d
	movl	%edx, %ecx
	movl	%eax, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-728(%rbp), %eax
	cltq
	addq	$40, %rax
	movl	-688(%rbp,%rax,4), %edi
	movl	-728(%rbp), %eax
	cltq
	addq	$32, %rax
	movl	-676(%rbp,%rax,4), %edx
	movl	-728(%rbp), %eax
	cltq
	addq	$28, %rax
	movl	-680(%rbp,%rax,4), %eax
	movl	-728(%rbp), %ecx
	leal	1(%rcx), %esi
	movl	%edi, %r8d
	movl	%edx, %ecx
	movl	%eax, %edx
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-688(%rbp), %rcx
	movl	-728(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rcx, %rax
	leaq	14(%rax), %rcx
	leaq	-688(%rbp), %rsi
	movl	-728(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rsi, %rax
	leaq	4(%rax), %rdx
	movl	-728(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -728(%rbp)
	movq	$28, -704(%rbp)
	jmp	.L43
.L20:
	subl	$1, -728(%rbp)
	movq	$0, -704(%rbp)
	jmp	.L43
.L33:
	movq	-752(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -696(%rbp)
	movq	-696(%rbp), %rax
	movq	%rax, -720(%rbp)
	movq	$8, -704(%rbp)
	jmp	.L43
.L36:
	movl	-728(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rbp, %rax
	subq	$508, %rax
	movzbl	(%rax), %eax
	cmpb	$36, %al
	jne	.L51
	movq	$32, -704(%rbp)
	jmp	.L43
.L51:
	movq	$0, -704(%rbp)
	jmp	.L43
.L25:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$25, -704(%rbp)
	jmp	.L43
.L29:
	movq	-720(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$23, -704(%rbp)
	jmp	.L43
.L24:
	movl	-632(%rbp), %eax
	cmpl	%eax, -728(%rbp)
	jge	.L53
	movq	$19, -704(%rbp)
	jmp	.L43
.L53:
	movq	$25, -704(%rbp)
	jmp	.L43
.L37:
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$12, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$20, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$24, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	4(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movl	-728(%rbp), %eax
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rcx
	movl	-728(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rcx, %rax
	leaq	4(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fread@PLT
	leaq	-688(%rbp), %rcx
	movl	-728(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rcx, %rax
	leaq	14(%rax), %rdi
	movq	-720(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fread@PLT
	movq	$6, -704(%rbp)
	jmp	.L43
.L40:
	addl	$1, -728(%rbp)
	movq	$24, -704(%rbp)
	jmp	.L43
.L58:
	nop
.L43:
	jmp	.L55
.L59:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L57
	call	__stack_chk_fail@PLT
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	imprimir_emprestimo, .-imprimir_emprestimo
	.section	.rodata
.LC8:
	.string	"rb"
.LC9:
	.string	"Nome: %s\n"
.LC10:
	.string	"Numero de cadastro: %d\n"
.LC11:
	.string	"Data de nascimento: %d/%d/%d\n"
.LC12:
	.string	"Rua %s numero %d, %s"
.LC13:
	.string	"Bairro: %s\n"
.LC14:
	.string	"Cidade: %s\n"
.LC15:
	.string	"Estado: %s\n"
	.align 8
.LC16:
	.string	"Digite o nome a ser procurado: "
	.text
	.globl	imprimir_cadastro
	.type	imprimir_cadastro, @function
imprimir_cadastro:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$368, %rsp
	movq	%rdi, -360(%rbp)
	movq	%rsi, -368(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -320(%rbp)
.L83:
	cmpq	$18, -320(%rbp)
	ja	.L86
	movq	-320(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L63(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L63(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L63:
	.long	.L86-.L63
	.long	.L74-.L63
	.long	.L73-.L63
	.long	.L72-.L63
	.long	.L71-.L63
	.long	.L70-.L63
	.long	.L86-.L63
	.long	.L87-.L63
	.long	.L86-.L63
	.long	.L86-.L63
	.long	.L68-.L63
	.long	.L67-.L63
	.long	.L86-.L63
	.long	.L86-.L63
	.long	.L66-.L63
	.long	.L65-.L63
	.long	.L64-.L63
	.long	.L86-.L63
	.long	.L62-.L63
	.text
.L62:
	movq	-360(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -312(%rbp)
	movq	-312(%rbp), %rax
	movq	%rax, -336(%rbp)
	movq	$4, -320(%rbp)
	jmp	.L75
.L71:
	cmpq	$0, -336(%rbp)
	jne	.L76
	movq	$14, -320(%rbp)
	jmp	.L75
.L76:
	movq	$10, -320(%rbp)
	jmp	.L75
.L66:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -320(%rbp)
	jmp	.L75
.L65:
	cmpl	$0, -340(%rbp)
	jne	.L78
	movq	$3, -320(%rbp)
	jmp	.L75
.L78:
	movq	$16, -320(%rbp)
	jmp	.L75
.L74:
	cmpq	$0, -328(%rbp)
	jne	.L80
	movq	$7, -320(%rbp)
	jmp	.L75
.L80:
	movq	$11, -320(%rbp)
	jmp	.L75
.L72:
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	52(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	60(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	64(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	68(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	72(%rdx), %rdi
	movq	%rax, %rcx
	movl	$40, %edx
	movl	$1, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	112(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	116(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$20, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	136(%rdx), %rdi
	movq	%rax, %rcx
	movl	$40, %edx
	movl	$1, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	176(%rdx), %rdi
	movq	%rax, %rcx
	movl	$30, %edx
	movl	$1, %esi
	call	fread@PLT
	movq	-336(%rbp), %rax
	leaq	-304(%rbp), %rdx
	leaq	206(%rdx), %rdi
	movq	%rax, %rcx
	movl	$20, %edx
	movl	$1, %esi
	call	fread@PLT
	leaq	-304(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-252(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-236(%rbp), %ecx
	movl	-240(%rbp), %edx
	movl	-244(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-192(%rbp), %eax
	leaq	-304(%rbp), %rdx
	addq	$116, %rdx
	leaq	-304(%rbp), %rcx
	leaq	72(%rcx), %rsi
	movq	%rdx, %rcx
	movl	%eax, %edx
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-304(%rbp), %rax
	addq	$136, %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-304(%rbp), %rax
	addq	$176, %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-304(%rbp), %rax
	addq	$206, %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-336(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-368(%rbp), %rdx
	leaq	-304(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	imprimir_emprestimo
	movq	$7, -320(%rbp)
	jmp	.L75
.L64:
	movq	-336(%rbp), %rax
	movl	$1, %edx
	movl	$174, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-336(%rbp), %rdx
	leaq	-341(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -328(%rbp)
	movq	$1, -320(%rbp)
	jmp	.L75
.L67:
	movq	-336(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$5, -320(%rbp)
	jmp	.L75
.L70:
	movq	-336(%rbp), %rdx
	leaq	-304(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fread@PLT
	leaq	-304(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -340(%rbp)
	movq	$15, -320(%rbp)
	jmp	.L75
.L68:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$5, -320(%rbp)
	jmp	.L75
.L73:
	movq	$18, -320(%rbp)
	jmp	.L75
.L86:
	nop
.L75:
	jmp	.L83
.L87:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L85
	call	__stack_chk_fail@PLT
.L85:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	imprimir_cadastro, .-imprimir_cadastro
	.section	.rodata
.LC17:
	.string	"Digite sua opcao: "
.LC18:
	.string	"\n\tBIBLIOTECA"
.LC19:
	.string	"\t>>Menu<<"
.LC20:
	.string	"1. Novo cadastro"
.LC21:
	.string	"2. Emprestimo"
.LC22:
	.string	"3. Busca de historico"
.LC23:
	.string	"4. Devolver Livro"
.LC24:
	.string	"5. Sair"
	.text
	.globl	menu
	.type	menu, @function
menu:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$10, -24(%rbp)
.L121:
	cmpq	$20, -24(%rbp)
	ja	.L124
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L91(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L91(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L91:
	.long	.L108-.L91
	.long	.L107-.L91
	.long	.L106-.L91
	.long	.L105-.L91
	.long	.L104-.L91
	.long	.L124-.L91
	.long	.L124-.L91
	.long	.L103-.L91
	.long	.L102-.L91
	.long	.L101-.L91
	.long	.L100-.L91
	.long	.L99-.L91
	.long	.L98-.L91
	.long	.L97-.L91
	.long	.L96-.L91
	.long	.L95-.L91
	.long	.L94-.L91
	.long	.L93-.L91
	.long	.L92-.L91
	.long	.L124-.L91
	.long	.L90-.L91
	.text
.L92:
	movzbl	-11(%rbp), %eax
	cmpb	$51, %al
	jne	.L109
	movq	$9, -24(%rbp)
	jmp	.L111
.L109:
	movq	$4, -24(%rbp)
	jmp	.L111
.L104:
	movzbl	-11(%rbp), %eax
	cmpb	$52, %al
	jne	.L112
	movq	$17, -24(%rbp)
	jmp	.L111
.L112:
	movq	$15, -24(%rbp)
	jmp	.L111
.L96:
	movl	-28(%rbp), %eax
	jmp	.L122
.L95:
	movzbl	-11(%rbp), %eax
	cmpb	$53, %al
	jne	.L115
	movq	$1, -24(%rbp)
	jmp	.L111
.L115:
	movq	$12, -24(%rbp)
	jmp	.L111
.L98:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-11(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$3, -24(%rbp)
	jmp	.L111
.L102:
	leaq	-11(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L111
.L107:
	leaq	-11(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movq	$14, -24(%rbp)
	jmp	.L111
.L105:
	movzbl	-11(%rbp), %eax
	cmpb	$49, %al
	jne	.L117
	movq	$0, -24(%rbp)
	jmp	.L111
.L117:
	movq	$2, -24(%rbp)
	jmp	.L111
.L94:
	movl	-28(%rbp), %eax
	jmp	.L122
.L99:
	movl	-28(%rbp), %eax
	jmp	.L122
.L101:
	leaq	-11(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movq	$20, -24(%rbp)
	jmp	.L111
.L97:
	movl	-28(%rbp), %eax
	jmp	.L122
.L93:
	leaq	-11(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L111
.L100:
	movq	$7, -24(%rbp)
	jmp	.L111
.L108:
	leaq	-11(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movq	$16, -24(%rbp)
	jmp	.L111
.L103:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -24(%rbp)
	jmp	.L111
.L106:
	movzbl	-11(%rbp), %eax
	cmpb	$50, %al
	jne	.L119
	movq	$8, -24(%rbp)
	jmp	.L111
.L119:
	movq	$18, -24(%rbp)
	jmp	.L111
.L90:
	movl	-28(%rbp), %eax
	jmp	.L122
.L124:
	nop
.L111:
	jmp	.L121
.L122:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L123
	call	__stack_chk_fail@PLT
.L123:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	menu, .-menu
	.section	.rodata
	.align 8
.LC25:
	.string	"Erro ao abrir o arquivo de emprestimo"
	.text
	.globl	verifica_emprestimo
	.type	verifica_emprestimo, @function
verifica_emprestimo:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$752, %rsp
	movq	%rdi, -744(%rbp)
	movq	%rsi, -752(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -704(%rbp)
.L149:
	cmpq	$17, -704(%rbp)
	ja	.L152
	movq	-704(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L128(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L128(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L128:
	.long	.L140-.L128
	.long	.L139-.L128
	.long	.L138-.L128
	.long	.L152-.L128
	.long	.L137-.L128
	.long	.L152-.L128
	.long	.L136-.L128
	.long	.L152-.L128
	.long	.L135-.L128
	.long	.L134-.L128
	.long	.L152-.L128
	.long	.L133-.L128
	.long	.L152-.L128
	.long	.L132-.L128
	.long	.L131-.L128
	.long	.L130-.L128
	.long	.L129-.L128
	.long	.L127-.L128
	.text
.L137:
	movq	-720(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$15, -704(%rbp)
	jmp	.L141
.L131:
	movl	$1, %eax
	jmp	.L150
.L130:
	movq	-720(%rbp), %rdx
	leaq	-688(%rbp), %rax
	movq	%rdx, %rcx
	movl	$50, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	leaq	-688(%rbp), %rdx
	movq	-744(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -724(%rbp)
	movq	$2, -704(%rbp)
	jmp	.L141
.L135:
	movl	$0, %eax
	jmp	.L150
.L139:
	movl	$-1, %eax
	jmp	.L150
.L129:
	cmpq	$0, -712(%rbp)
	jne	.L143
	movq	$8, -704(%rbp)
	jmp	.L141
.L143:
	movq	$4, -704(%rbp)
	jmp	.L141
.L133:
	movq	-752(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -696(%rbp)
	movq	-696(%rbp), %rax
	movq	%rax, -720(%rbp)
	movq	$0, -704(%rbp)
	jmp	.L141
.L134:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -704(%rbp)
	jmp	.L141
.L132:
	movq	-720(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$14, -704(%rbp)
	jmp	.L141
.L127:
	movq	-720(%rbp), %rax
	movl	$1, %edx
	movl	$628, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-720(%rbp), %rdx
	leaq	-728(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -712(%rbp)
	movq	$16, -704(%rbp)
	jmp	.L141
.L136:
	movq	$11, -704(%rbp)
	jmp	.L141
.L140:
	cmpq	$0, -720(%rbp)
	jne	.L145
	movq	$9, -704(%rbp)
	jmp	.L141
.L145:
	movq	$15, -704(%rbp)
	jmp	.L141
.L138:
	cmpl	$0, -724(%rbp)
	jne	.L147
	movq	$13, -704(%rbp)
	jmp	.L141
.L147:
	movq	$17, -704(%rbp)
	jmp	.L141
.L152:
	nop
.L141:
	jmp	.L149
.L150:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L151
	call	__stack_chk_fail@PLT
.L151:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	verifica_emprestimo, .-verifica_emprestimo
	.section	.rodata
.LC26:
	.string	"Digite o nome: "
	.align 8
.LC27:
	.string	"%d. Nome: %s   Autor: %s(%d/%d/%d)\n"
.LC28:
	.string	"Livro %d atrasado\n"
	.align 8
.LC29:
	.string	"Nao pode realizar emprestimos devido aos atrasos"
.LC30:
	.string	"Digite o nome do livro %d: "
	.align 8
.LC31:
	.string	"Digite o nome do autor do livro %d: "
	.align 8
.LC32:
	.string	"Digite a quantidade de livros: "
.LC33:
	.string	"%d"
	.align 8
.LC34:
	.string	"Erro: quantidade acima da permitida"
	.align 8
.LC35:
	.string	"Digite a quantidade de livros (max %d): "
.LC36:
	.string	"Escolha o livro devolvido:"
	.text
	.globl	altera_emprestimo
	.type	altera_emprestimo, @function
altera_emprestimo:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1312, %rsp
	movl	%edi, -1284(%rbp)
	movl	%esi, -1288(%rbp)
	movl	%edx, -1292(%rbp)
	movq	%rcx, -1304(%rbp)
	movq	%r8, -1312(%rbp)
	movl	%r9d, -1296(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$88, -1216(%rbp)
.L264:
	cmpq	$104, -1216(%rbp)
	ja	.L267
	movq	-1216(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L156(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L156(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L156:
	.long	.L267-.L156
	.long	.L215-.L156
	.long	.L267-.L156
	.long	.L214-.L156
	.long	.L213-.L156
	.long	.L267-.L156
	.long	.L212-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L211-.L156
	.long	.L210-.L156
	.long	.L209-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L208-.L156
	.long	.L207-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L206-.L156
	.long	.L205-.L156
	.long	.L267-.L156
	.long	.L204-.L156
	.long	.L267-.L156
	.long	.L203-.L156
	.long	.L202-.L156
	.long	.L267-.L156
	.long	.L201-.L156
	.long	.L200-.L156
	.long	.L199-.L156
	.long	.L198-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L197-.L156
	.long	.L196-.L156
	.long	.L195-.L156
	.long	.L194-.L156
	.long	.L193-.L156
	.long	.L192-.L156
	.long	.L191-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L190-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L189-.L156
	.long	.L188-.L156
	.long	.L187-.L156
	.long	.L186-.L156
	.long	.L267-.L156
	.long	.L185-.L156
	.long	.L184-.L156
	.long	.L183-.L156
	.long	.L267-.L156
	.long	.L182-.L156
	.long	.L181-.L156
	.long	.L180-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L179-.L156
	.long	.L178-.L156
	.long	.L177-.L156
	.long	.L176-.L156
	.long	.L267-.L156
	.long	.L175-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L174-.L156
	.long	.L173-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L172-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L171-.L156
	.long	.L170-.L156
	.long	.L169-.L156
	.long	.L168-.L156
	.long	.L167-.L156
	.long	.L166-.L156
	.long	.L267-.L156
	.long	.L165-.L156
	.long	.L164-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L163-.L156
	.long	.L268-.L156
	.long	.L161-.L156
	.long	.L160-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L159-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L158-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L267-.L156
	.long	.L157-.L156
	.long	.L267-.L156
	.long	.L155-.L156
	.text
.L185:
	cmpl	$4, -1248(%rbp)
	jg	.L216
	movq	$54, -1216(%rbp)
	jmp	.L218
.L216:
	movq	$55, -1216(%rbp)
	jmp	.L218
.L168:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$12, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rcx, %rax
	leaq	4(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fread@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$39, -1216(%rbp)
	jmp	.L218
.L155:
	movl	-712(%rbp), %edx
	movl	-1268(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -712(%rbp)
	movl	-712(%rbp), %eax
	negl	%eax
	cltq
	imulq	$524, %rax, %rax
	movq	%rax, %rcx
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-4, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	leaq	-768(%rbp), %rdx
	leaq	56(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	$81, -1216(%rbp)
	jmp	.L218
.L202:
	movq	-1232(%rbp), %rdx
	leaq	-1284(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1288(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1292(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	leaq	-1184(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	(%rcx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fwrite@PLT
	leaq	-976(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	(%rcx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fwrite@PLT
	addl	$1, -1248(%rbp)
	movq	$62, -1216(%rbp)
	jmp	.L218
.L183:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	addl	$1, -1248(%rbp)
	movq	$27, -1216(%rbp)
	jmp	.L218
.L213:
	addl	$1, -1248(%rbp)
	movq	$47, -1216(%rbp)
	jmp	.L218
.L198:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-74(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-74(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -1252(%rbp)
	movl	-712(%rbp), %eax
	negl	%eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$5, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rcx
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-4, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movl	-712(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -712(%rbp)
	movq	-1232(%rbp), %rax
	leaq	-768(%rbp), %rdx
	leaq	56(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-1252(%rbp), %eax
	subl	$1, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$5, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rcx
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$24, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1270(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$81, -1216(%rbp)
	jmp	.L218
.L176:
	movl	-1268(%rbp), %eax
	cmpl	%eax, -1248(%rbp)
	jge	.L219
	movq	$25, -1216(%rbp)
	jmp	.L218
.L219:
	movq	$104, -1216(%rbp)
	jmp	.L218
.L157:
	cmpl	$1, -1296(%rbp)
	jne	.L221
	movq	$56, -1216(%rbp)
	jmp	.L218
.L221:
	movq	$35, -1216(%rbp)
	jmp	.L218
.L208:
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-1240(%rbp), %rax
	movl	(%rax), %eax
	leaq	-64(%rbp), %rdx
	movq	-1312(%rbp), %rsi
	movl	$2, %ecx
	movl	%eax, %edi
	call	confirma_senha
	movq	$9, -1216(%rbp)
	jmp	.L218
.L166:
	movq	-1232(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -1208(%rbp)
	movl	-1248(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-1240(%rbp), %rax
	addq	%rdx, %rax
	movq	-1208(%rbp), %rdx
	movl	%edx, (%rax)
	movq	$29, -1216(%rbp)
	jmp	.L218
.L180:
	movl	$0, -1252(%rbp)
	movl	$0, -1248(%rbp)
	movq	$27, -1216(%rbp)
	jmp	.L218
.L169:
	cmpl	$0, -1244(%rbp)
	jne	.L224
	movq	$61, -1216(%rbp)
	jmp	.L218
.L224:
	movq	$34, -1216(%rbp)
	jmp	.L218
.L174:
	movl	-712(%rbp), %eax
	negl	%eax
	cltq
	imulq	$524, %rax, %rax
	movq	%rax, %rcx
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$24, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movl	$0, -1248(%rbp)
	movq	$50, -1216(%rbp)
	jmp	.L218
.L189:
	movl	$0, -1248(%rbp)
	movq	$77, -1216(%rbp)
	jmp	.L218
.L182:
	movq	-1232(%rbp), %rdx
	leaq	-1269(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$33, -1216(%rbp)
	jmp	.L218
.L170:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$81, -1216(%rbp)
	jmp	.L218
.L215:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$12, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rdx
	movl	-1248(%rbp), %eax
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rcx, %rax
	leaq	4(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fread@PLT
	leaq	-768(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rcx, %rax
	leaq	14(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fread@PLT
	movl	-1248(%rbp), %eax
	cltq
	addq	$40, %rax
	movl	-768(%rbp,%rax,4), %esi
	movl	-1248(%rbp), %eax
	cltq
	addq	$32, %rax
	movl	-756(%rbp,%rax,4), %r8d
	movl	-1248(%rbp), %eax
	cltq
	addq	$28, %rax
	movl	-760(%rbp,%rax,4), %edi
	leaq	-768(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rcx, %rax
	leaq	14(%rax), %rcx
	leaq	-768(%rbp), %r9
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%r9, %rax
	leaq	4(%rax), %rdx
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	subq	$8, %rsp
	pushq	%rsi
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %esi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	addl	$1, -1248(%rbp)
	movq	$28, -1216(%rbp)
	jmp	.L218
.L167:
	movq	-1232(%rbp), %rdx
	leaq	-768(%rbp), %rax
	movq	%rdx, %rcx
	movl	$50, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	leaq	-768(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1244(%rbp)
	movq	$79, -1216(%rbp)
	jmp	.L218
.L171:
	movl	-1268(%rbp), %eax
	cmpl	%eax, -1248(%rbp)
	jge	.L226
	movq	$19, -1216(%rbp)
	jmp	.L218
.L226:
	movq	$84, -1216(%rbp)
	jmp	.L218
.L173:
	movb	$36, -1270(%rbp)
	movq	-1304(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1200(%rbp)
	movq	-1200(%rbp), %rax
	movq	%rax, -1232(%rbp)
	movl	$4, %esi
	movl	$5, %edi
	call	calloc@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movq	%rax, -1240(%rbp)
	movq	$38, -1216(%rbp)
	jmp	.L218
.L214:
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-1252(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1252(%rbp)
	movq	$51, -1216(%rbp)
	jmp	.L218
.L207:
	movq	-1232(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$89, -1216(%rbp)
	jmp	.L218
.L203:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -1216(%rbp)
	jmp	.L218
.L159:
	movl	-1248(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-1240(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movslq	%eax, %rcx
	movq	-1232(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movq	$-24, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1284(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1288(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1292(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-712(%rbp), %edx
	movl	-1248(%rbp), %eax
	addl	%edx, %eax
	leaq	-768(%rbp), %rdx
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	leaq	-1184(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	(%rcx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fwrite@PLT
	leaq	-976(%rbp), %rcx
	movl	-1248(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	(%rcx,%rax), %rdi
	movq	-1232(%rbp), %rax
	movq	%rax, %rcx
	movl	$50, %edx
	movl	$1, %esi
	call	fwrite@PLT
	movq	$4, -1216(%rbp)
	jmp	.L218
.L194:
	movl	-1248(%rbp), %eax
	cltq
	addq	$40, %rax
	movl	-768(%rbp,%rax,4), %edx
	movl	-1292(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L228
	movq	$91, -1216(%rbp)
	jmp	.L218
.L228:
	movq	$52, -1216(%rbp)
	jmp	.L218
.L164:
	cmpl	$1, -1296(%rbp)
	jne	.L230
	movq	$22, -1216(%rbp)
	jmp	.L218
.L230:
	movq	$45, -1216(%rbp)
	jmp	.L218
.L209:
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-1252(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1252(%rbp)
	movq	$36, -1216(%rbp)
	jmp	.L218
.L211:
	cmpl	$1, -1296(%rbp)
	jne	.L232
	movq	$74, -1216(%rbp)
	jmp	.L218
.L232:
	movq	$6, -1216(%rbp)
	jmp	.L218
.L184:
	movl	-1248(%rbp), %eax
	cltq
	addq	$32, %rax
	movl	-756(%rbp,%rax,4), %edx
	movl	-1288(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L234
	movq	$64, -1216(%rbp)
	jmp	.L218
.L234:
	movq	$52, -1216(%rbp)
	jmp	.L218
.L206:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-1184(%rbp), %rsi
	movl	-1248(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	addq	%rcx, %rax
	addq	%rax, %rax
	addq	%rsi, %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-976(%rbp), %rsi
	movl	-1248(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	addq	%rcx, %rax
	addq	%rax, %rax
	addq	%rsi, %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	addl	$1, -1248(%rbp)
	movq	$77, -1216(%rbp)
	jmp	.L218
.L161:
	movl	-1248(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-1252(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -1252(%rbp)
	movq	$52, -1216(%rbp)
	jmp	.L218
.L181:
	movl	$0, -1248(%rbp)
	movq	$47, -1216(%rbp)
	jmp	.L218
.L178:
	cmpq	$0, -1224(%rbp)
	jne	.L236
	movq	$16, -1216(%rbp)
	jmp	.L218
.L236:
	movq	$78, -1216(%rbp)
	jmp	.L218
.L179:
	movl	-712(%rbp), %eax
	cmpl	$4, %eax
	jg	.L238
	movq	$42, -1216(%rbp)
	jmp	.L218
.L238:
	movq	$69, -1216(%rbp)
	jmp	.L218
.L212:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	$81, -1216(%rbp)
	jmp	.L218
.L201:
	movl	-712(%rbp), %eax
	cmpl	%eax, -1248(%rbp)
	jge	.L240
	movq	$80, -1216(%rbp)
	jmp	.L218
.L240:
	movq	$46, -1216(%rbp)
	jmp	.L218
.L192:
	cmpq	$0, -1232(%rbp)
	jne	.L242
	movq	$48, -1216(%rbp)
	jmp	.L218
.L242:
	movq	$15, -1216(%rbp)
	jmp	.L218
.L177:
	movq	-1232(%rbp), %rax
	leaq	-768(%rbp), %rdx
	leaq	52(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	-1232(%rbp), %rax
	leaq	-768(%rbp), %rdx
	leaq	56(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	$102, -1216(%rbp)
	jmp	.L218
.L165:
	movl	-1292(%rbp), %edx
	movl	-1288(%rbp), %esi
	movl	-1284(%rbp), %eax
	leaq	-1256(%rbp), %r8
	leaq	-1260(%rbp), %rdi
	leaq	-1264(%rbp), %rcx
	movq	%r8, %r9
	movq	%rdi, %r8
	movl	%eax, %edi
	call	devolucao
	movl	-712(%rbp), %eax
	movl	%eax, -1248(%rbp)
	movq	$97, -1216(%rbp)
	jmp	.L218
.L196:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$628, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1232(%rbp), %rdx
	leaq	-1252(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -1224(%rbp)
	movq	$60, -1216(%rbp)
	jmp	.L218
.L172:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	-1268(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -1216(%rbp)
	jmp	.L218
.L186:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$89, -1216(%rbp)
	jmp	.L218
.L204:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-712(%rbp), %edx
	movl	$5, %eax
	subl	%edx, %eax
	movl	%eax, %esi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1268(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$20, -1216(%rbp)
	jmp	.L218
.L200:
	movl	-712(%rbp), %eax
	cmpl	%eax, -1248(%rbp)
	jge	.L244
	movq	$1, -1216(%rbp)
	jmp	.L218
.L244:
	movq	$30, -1216(%rbp)
	jmp	.L218
.L187:
	cmpl	$4, -1248(%rbp)
	jg	.L246
	movq	$10, -1216(%rbp)
	jmp	.L218
.L246:
	movq	$104, -1216(%rbp)
	jmp	.L218
.L160:
	movl	-1248(%rbp), %eax
	cltq
	addq	$32, %rax
	movl	-756(%rbp,%rax,4), %edx
	movl	-1288(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L248
	movq	$3, -1216(%rbp)
	jmp	.L218
.L248:
	movq	$51, -1216(%rbp)
	jmp	.L218
.L158:
	cmpl	$4, -1248(%rbp)
	jg	.L250
	movq	$37, -1216(%rbp)
	jmp	.L218
.L250:
	movq	$59, -1216(%rbp)
	jmp	.L218
.L197:
	movzbl	-1269(%rbp), %edx
	movzbl	-1270(%rbp), %eax
	cmpb	%al, %dl
	jne	.L252
	movq	$82, -1216(%rbp)
	jmp	.L218
.L252:
	movq	$29, -1216(%rbp)
	jmp	.L218
.L193:
	movl	-1264(%rbp), %eax
	movl	-1248(%rbp), %edx
	movslq	%edx, %rdx
	addq	$28, %rdx
	movl	%eax, -760(%rbp,%rdx,4)
	movl	-1260(%rbp), %eax
	movl	-1248(%rbp), %edx
	movslq	%edx, %rdx
	addq	$32, %rdx
	movl	%eax, -756(%rbp,%rdx,4)
	movl	-1256(%rbp), %eax
	movl	-1248(%rbp), %edx
	movslq	%edx, %rdx
	addq	$40, %rdx
	movl	%eax, -768(%rbp,%rdx,4)
	addl	$1, -1248(%rbp)
	movq	$97, -1216(%rbp)
	jmp	.L218
.L175:
	movl	-1248(%rbp), %eax
	cltq
	addq	$28, %rax
	movl	-760(%rbp,%rax,4), %edx
	movl	-1284(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L254
	movq	$90, -1216(%rbp)
	jmp	.L218
.L254:
	movq	$52, -1216(%rbp)
	jmp	.L218
.L210:
	movl	-1248(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-1240(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L256
	movq	$94, -1216(%rbp)
	jmp	.L218
.L256:
	movq	$4, -1216(%rbp)
	jmp	.L218
.L190:
	movl	$0, -1248(%rbp)
	movq	$62, -1216(%rbp)
	jmp	.L218
.L188:
	movl	-1252(%rbp), %eax
	testl	%eax, %eax
	je	.L258
	movq	$24, -1216(%rbp)
	jmp	.L218
.L258:
	movq	$20, -1216(%rbp)
	jmp	.L218
.L191:
	movl	-1248(%rbp), %eax
	cltq
	addq	$40, %rax
	movl	-768(%rbp,%rax,4), %edx
	movl	-1292(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L260
	movq	$11, -1216(%rbp)
	jmp	.L218
.L260:
	movq	$36, -1216(%rbp)
	jmp	.L218
.L163:
	movq	$70, -1216(%rbp)
	jmp	.L218
.L195:
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -1248(%rbp)
	movq	$28, -1216(%rbp)
	jmp	.L218
.L199:
	movq	-1232(%rbp), %rax
	movl	$1, %edx
	movl	$74, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	addl	$1, -1248(%rbp)
	movq	$50, -1216(%rbp)
	jmp	.L218
.L205:
	movl	-712(%rbp), %edx
	movl	-1268(%rbp), %eax
	addl	%edx, %eax
	cmpl	$5, %eax
	jle	.L262
	movq	$85, -1216(%rbp)
	jmp	.L218
.L262:
	movq	$45, -1216(%rbp)
	jmp	.L218
.L267:
	nop
.L218:
	jmp	.L264
.L268:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L266
	call	__stack_chk_fail@PLT
.L266:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	altera_emprestimo, .-altera_emprestimo
	.section	.rodata
.LC37:
	.string	"Data de hoje"
.LC38:
	.string	"Dia: "
.LC39:
	.string	"Mes: "
.LC40:
	.string	"Ano: "
	.align 8
.LC41:
	.string	"Nao existe cadastro nesse nome"
	.align 8
.LC42:
	.string	"Erro, nao foi possivel abrir o arquivo de cadastro"
.LC43:
	.string	"Nome: "
.LC44:
	.string	"Cadastro.bin"
.LC45:
	.string	"Emprestimo.bin"
	.align 8
.LC46:
	.string	"Erro, nao foi possivel abrir o arquivo de emprestimo"
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$424, %rsp
	.cfi_offset 3, -24
	movl	%edi, -404(%rbp)
	movq	%rsi, -416(%rbp)
	movq	%rdx, -424(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Jvlc_envp(%rip)
	nop
.L270:
	movq	$0, _TIG_IZ_Jvlc_argv(%rip)
	nop
.L271:
	movl	$0, _TIG_IZ_Jvlc_argc(%rip)
	nop
	nop
.L272:
.L273:
#APP
# 566 "murilocpontes_Trabalho-2-Lab-ICC-biblioteca_main_windows.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Jvlc--0
# 0 "" 2
#NO_APP
	movl	-404(%rbp), %eax
	movl	%eax, _TIG_IZ_Jvlc_argc(%rip)
	movq	-416(%rbp), %rax
	movq	%rax, _TIG_IZ_Jvlc_argv(%rip)
	movq	-424(%rbp), %rax
	movq	%rax, _TIG_IZ_Jvlc_envp(%rip)
	nop
	movq	$38, -336(%rbp)
.L335:
	cmpq	$46, -336(%rbp)
	ja	.L338
	movq	-336(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L276(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L276(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L276:
	.long	.L306-.L276
	.long	.L305-.L276
	.long	.L304-.L276
	.long	.L303-.L276
	.long	.L302-.L276
	.long	.L338-.L276
	.long	.L301-.L276
	.long	.L300-.L276
	.long	.L299-.L276
	.long	.L298-.L276
	.long	.L338-.L276
	.long	.L297-.L276
	.long	.L338-.L276
	.long	.L296-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L295-.L276
	.long	.L294-.L276
	.long	.L293-.L276
	.long	.L292-.L276
	.long	.L291-.L276
	.long	.L290-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L289-.L276
	.long	.L288-.L276
	.long	.L338-.L276
	.long	.L287-.L276
	.long	.L286-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L285-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L284-.L276
	.long	.L338-.L276
	.long	.L283-.L276
	.long	.L282-.L276
	.long	.L281-.L276
	.long	.L280-.L276
	.long	.L279-.L276
	.long	.L338-.L276
	.long	.L338-.L276
	.long	.L278-.L276
	.long	.L277-.L276
	.long	.L275-.L276
	.text
.L293:
	movl	$-1, %eax
	jmp	.L336
.L289:
	movq	-344(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-360(%rbp), %rax
	movq	%rax, %rdi
	call	verifica_id
	movl	%eax, -400(%rbp)
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-85(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-85(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -384(%rbp)
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-85(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-85(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -380(%rbp)
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-85(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-85(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -376(%rbp)
	movq	$26, -336(%rbp)
	jmp	.L308
.L302:
	movl	$0, %eax
	jmp	.L336
.L299:
	movq	-344(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-352(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -344(%rbp)
	movq	$0, -336(%rbp)
	jmp	.L308
.L277:
	cmpl	$0, -368(%rbp)
	jne	.L309
	movq	$37, -336(%rbp)
	jmp	.L308
.L309:
	movq	$6, -336(%rbp)
	jmp	.L308
.L305:
	cmpl	$5, -388(%rbp)
	ja	.L311
	movl	-388(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L313(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L313(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L313:
	.long	.L311-.L313
	.long	.L317-.L313
	.long	.L316-.L313
	.long	.L315-.L313
	.long	.L314-.L313
	.long	.L312-.L313
	.text
.L312:
	movq	$4, -336(%rbp)
	jmp	.L318
.L314:
	movq	$29, -336(%rbp)
	jmp	.L318
.L315:
	movq	$17, -336(%rbp)
	jmp	.L318
.L316:
	movq	$13, -336(%rbp)
	jmp	.L318
.L317:
	movq	$44, -336(%rbp)
	jmp	.L318
.L311:
	movq	$26, -336(%rbp)
	nop
.L318:
	jmp	.L308
.L303:
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -336(%rbp)
	jmp	.L308
.L295:
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$41, -336(%rbp)
	jmp	.L308
.L290:
	cmpl	$0, -392(%rbp)
	jne	.L319
	movq	$7, -336(%rbp)
	jmp	.L308
.L319:
	movq	$26, -336(%rbp)
	jmp	.L308
.L288:
	call	menu
	movl	%eax, -388(%rbp)
	movq	$1, -336(%rbp)
	jmp	.L308
.L297:
	cmpl	$0, -396(%rbp)
	je	.L321
	movq	$32, -336(%rbp)
	jmp	.L308
.L321:
	movq	$39, -336(%rbp)
	jmp	.L308
.L298:
	cmpl	$1, -372(%rbp)
	jne	.L323
	movq	$19, -336(%rbp)
	jmp	.L308
.L323:
	movq	$26, -336(%rbp)
	jmp	.L308
.L296:
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-80(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-360(%rbp), %rcx
	leaq	-80(%rbp), %rdx
	movl	-400(%rbp), %esi
	leaq	-320(%rbp), %rax
	movq	%rax, %rdi
	call	verifica_cad
	movl	%eax, -396(%rbp)
	movq	$11, -336(%rbp)
	jmp	.L308
.L292:
	movq	-360(%rbp), %rdi
	movq	-352(%rbp), %rcx
	movl	-376(%rbp), %edx
	movl	-380(%rbp), %esi
	movl	-384(%rbp), %eax
	movl	$1, %r9d
	movq	%rdi, %r8
	movl	%eax, %edi
	call	altera_emprestimo
	movq	$26, -336(%rbp)
	jmp	.L308
.L285:
	cmpl	$-1, -396(%rbp)
	je	.L325
	movq	$46, -336(%rbp)
	jmp	.L308
.L325:
	movq	$39, -336(%rbp)
	jmp	.L308
.L294:
	movq	-352(%rbp), %rdx
	movq	-360(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	imprimir_cadastro
	movq	$26, -336(%rbp)
	jmp	.L308
.L280:
	movq	-352(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	verifica_emprestimo
	movl	%eax, -368(%rbp)
	movq	$45, -336(%rbp)
	jmp	.L308
.L301:
	movq	-352(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	verifica_emprestimo
	movl	%eax, -372(%rbp)
	movq	$9, -336(%rbp)
	jmp	.L308
.L282:
	movq	$28, -336(%rbp)
	jmp	.L308
.L287:
	leaq	.LC44(%rip), %rax
	movq	%rax, -360(%rbp)
	leaq	.LC45(%rip), %rax
	movq	%rax, -352(%rbp)
	movl	$0, -400(%rbp)
	movq	-360(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -328(%rbp)
	movq	-328(%rbp), %rax
	movq	%rax, -344(%rbp)
	movq	$20, -336(%rbp)
	jmp	.L308
.L278:
	movq	-360(%rbp), %rdx
	movl	-400(%rbp), %ecx
	leaq	-320(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	cadastro
	movl	%eax, -392(%rbp)
	movq	$21, -336(%rbp)
	jmp	.L308
.L283:
	movq	-352(%rbp), %r9
	movq	-360(%rbp), %r8
	movl	-376(%rbp), %r10d
	movl	-380(%rbp), %edx
	movl	-384(%rbp), %esi
	movl	-396(%rbp), %edi
	subq	$8, %rsp
	subq	$232, %rsp
	movq	%rsp, %rax
	movq	-320(%rbp), %rcx
	movq	-312(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-304(%rbp), %rcx
	movq	-296(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-288(%rbp), %rcx
	movq	-280(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-272(%rbp), %rcx
	movq	-264(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-256(%rbp), %rcx
	movq	-248(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-240(%rbp), %rcx
	movq	-232(%rbp), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	-224(%rbp), %rcx
	movq	-216(%rbp), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	-208(%rbp), %rcx
	movq	-200(%rbp), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-192(%rbp), %rcx
	movq	-184(%rbp), %rbx
	movq	%rcx, 128(%rax)
	movq	%rbx, 136(%rax)
	movq	-176(%rbp), %rcx
	movq	-168(%rbp), %rbx
	movq	%rcx, 144(%rax)
	movq	%rbx, 152(%rax)
	movq	-160(%rbp), %rcx
	movq	-152(%rbp), %rbx
	movq	%rcx, 160(%rax)
	movq	%rbx, 168(%rax)
	movq	-144(%rbp), %rcx
	movq	-136(%rbp), %rbx
	movq	%rcx, 176(%rax)
	movq	%rbx, 184(%rax)
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, 192(%rax)
	movq	%rbx, 200(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 208(%rax)
	movq	%rbx, 216(%rax)
	movl	-96(%rbp), %ecx
	movl	%ecx, 224(%rax)
	movl	%r10d, %ecx
	call	novo_emprestimo
	addq	$240, %rsp
	movq	$26, -336(%rbp)
	jmp	.L308
.L279:
	movl	$-1, %eax
	jmp	.L336
.L306:
	cmpq	$0, -344(%rbp)
	jne	.L327
	movq	$35, -336(%rbp)
	jmp	.L308
.L327:
	movq	$25, -336(%rbp)
	jmp	.L308
.L275:
	movq	-360(%rbp), %rsi
	movl	-396(%rbp), %eax
	movl	$1, %ecx
	movl	$0, %edx
	movl	%eax, %edi
	call	confirma_senha
	movl	%eax, -364(%rbp)
	movq	$2, -336(%rbp)
	jmp	.L308
.L281:
	cmpl	$0, -396(%rbp)
	jne	.L329
	movq	$3, -336(%rbp)
	jmp	.L308
.L329:
	movq	$17, -336(%rbp)
	jmp	.L308
.L300:
	movq	-360(%rbp), %rdx
	leaq	-320(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	salva_cadastro
	addl	$1, -400(%rbp)
	movq	$26, -336(%rbp)
	jmp	.L308
.L284:
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -336(%rbp)
	jmp	.L308
.L286:
	movq	-360(%rbp), %rdi
	movq	-352(%rbp), %rcx
	movl	-376(%rbp), %edx
	movl	-380(%rbp), %esi
	movl	-384(%rbp), %eax
	movl	$2, %r9d
	movq	%rdi, %r8
	movl	%eax, %edi
	call	altera_emprestimo
	movq	$26, -336(%rbp)
	jmp	.L308
.L304:
	cmpl	$0, -364(%rbp)
	je	.L331
	movq	$40, -336(%rbp)
	jmp	.L308
.L331:
	movq	$26, -336(%rbp)
	jmp	.L308
.L291:
	cmpq	$0, -344(%rbp)
	jne	.L333
	movq	$16, -336(%rbp)
	jmp	.L308
.L333:
	movq	$8, -336(%rbp)
	jmp	.L308
.L338:
	nop
.L308:
	jmp	.L335
.L336:
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L337
	call	__stack_chk_fail@PLT
.L337:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.globl	verifica_cad
	.type	verifica_cad, @function
verifica_cad:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movl	%esi, -76(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%rcx, -96(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$17, -32(%rbp)
.L364:
	cmpq	$18, -32(%rbp)
	ja	.L367
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L342(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L342(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L342:
	.long	.L355-.L342
	.long	.L367-.L342
	.long	.L354-.L342
	.long	.L367-.L342
	.long	.L367-.L342
	.long	.L367-.L342
	.long	.L367-.L342
	.long	.L353-.L342
	.long	.L352-.L342
	.long	.L351-.L342
	.long	.L350-.L342
	.long	.L349-.L342
	.long	.L348-.L342
	.long	.L347-.L342
	.long	.L346-.L342
	.long	.L345-.L342
	.long	.L344-.L342
	.long	.L343-.L342
	.long	.L341-.L342
	.text
.L341:
	cmpq	$0, -48(%rbp)
	jne	.L356
	movq	$11, -32(%rbp)
	jmp	.L358
.L356:
	movq	$9, -32(%rbp)
	jmp	.L358
.L346:
	cmpq	$0, -40(%rbp)
	jne	.L359
	movq	$0, -32(%rbp)
	jmp	.L358
.L359:
	movq	$7, -32(%rbp)
	jmp	.L358
.L345:
	movq	-48(%rbp), %rax
	movl	$1, %edx
	movl	$174, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-48(%rbp), %rdx
	leaq	-60(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -40(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L358
.L348:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	%eax, -56(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$10, -32(%rbp)
	jmp	.L358
.L352:
	movq	-96(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L358
.L344:
	movl	$0, %eax
	jmp	.L365
.L349:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -32(%rbp)
	jmp	.L358
.L351:
	movq	-72(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, %rcx
	movl	$50, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-72(%rbp), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -52(%rbp)
	movq	$2, -32(%rbp)
	jmp	.L358
.L347:
	movl	$-1, %eax
	jmp	.L365
.L343:
	movq	$8, -32(%rbp)
	jmp	.L358
.L350:
	movl	-56(%rbp), %eax
	jmp	.L365
.L355:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$16, -32(%rbp)
	jmp	.L358
.L353:
	movq	-48(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$9, -32(%rbp)
	jmp	.L358
.L354:
	cmpl	$0, -52(%rbp)
	jne	.L362
	movq	$12, -32(%rbp)
	jmp	.L358
.L362:
	movq	$15, -32(%rbp)
	jmp	.L358
.L367:
	nop
.L358:
	jmp	.L364
.L365:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L366
	call	__stack_chk_fail@PLT
.L366:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	verifica_cad, .-verifica_cad
	.globl	verifica_id
	.type	verifica_id, @function
verifica_id:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -24(%rbp)
.L381:
	cmpq	$6, -24(%rbp)
	ja	.L384
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L371(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L371(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L371:
	.long	.L376-.L371
	.long	.L375-.L371
	.long	.L384-.L371
	.long	.L374-.L371
	.long	.L373-.L371
	.long	.L372-.L371
	.long	.L370-.L371
	.text
.L373:
	movq	-56(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rdx
	leaq	-45(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -32(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L377
.L375:
	movq	-40(%rbp), %rax
	movl	$2, %edx
	movq	$-174, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-40(%rbp), %rdx
	leaq	-44(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	$5, -24(%rbp)
	jmp	.L377
.L374:
	cmpq	$0, -32(%rbp)
	jne	.L378
	movq	$0, -24(%rbp)
	jmp	.L377
.L378:
	movq	$1, -24(%rbp)
	jmp	.L377
.L370:
	movq	$4, -24(%rbp)
	jmp	.L377
.L372:
	movl	-44(%rbp), %eax
	subl	$99999, %eax
	jmp	.L382
.L376:
	movl	$0, %eax
	jmp	.L382
.L384:
	nop
.L377:
	jmp	.L381
.L382:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L383
	call	__stack_chk_fail@PLT
.L383:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	verifica_id, .-verifica_id
	.section	.rodata
	.align 8
.LC47:
	.string	"Digite a quantidade de livros a serem emprestados: "
	.align 8
.LC48:
	.string	"Erro: quantidade de livros acima da permitida"
	.align 8
.LC49:
	.string	"Erro ao abrir o arquivo de cad"
.LC50:
	.string	"Digite o nome do autor: "
	.align 8
.LC51:
	.string	"Erro ao abrir o arquivo de empr"
	.text
	.globl	novo_emprestimo
	.type	novo_emprestimo, @function
novo_emprestimo:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$768, %rsp
	movl	%edi, -740(%rbp)
	movl	%esi, -744(%rbp)
	movl	%edx, -748(%rbp)
	movl	%ecx, -752(%rbp)
	movq	%r8, -760(%rbp)
	movq	%r9, -768(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$33, -704(%rbp)
.L424:
	cmpq	$37, -704(%rbp)
	ja	.L427
	movq	-704(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L388(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L388(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L388:
	.long	.L427-.L388
	.long	.L409-.L388
	.long	.L427-.L388
	.long	.L408-.L388
	.long	.L407-.L388
	.long	.L427-.L388
	.long	.L406-.L388
	.long	.L405-.L388
	.long	.L404-.L388
	.long	.L427-.L388
	.long	.L403-.L388
	.long	.L427-.L388
	.long	.L402-.L388
	.long	.L401-.L388
	.long	.L400-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L399-.L388
	.long	.L398-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L397-.L388
	.long	.L396-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L427-.L388
	.long	.L395-.L388
	.long	.L394-.L388
	.long	.L393-.L388
	.long	.L392-.L388
	.long	.L391-.L388
	.long	.L390-.L388
	.long	.L389-.L388
	.long	.L428-.L388
	.text
.L397:
	movq	-760(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -696(%rbp)
	movq	-696(%rbp), %rax
	movq	%rax, -712(%rbp)
	movq	$35, -704(%rbp)
	jmp	.L410
.L407:
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-688(%rbp), %rax
	addq	$56, %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	$13, -704(%rbp)
	jmp	.L410
.L395:
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-688(%rbp), %rax
	addq	$56, %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -704(%rbp)
	jmp	.L410
.L400:
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -704(%rbp)
	jmp	.L410
.L394:
	movq	-712(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$37, -704(%rbp)
	jmp	.L410
.L402:
	movl	-632(%rbp), %eax
	cmpl	%eax, -716(%rbp)
	jge	.L411
	movq	$6, -704(%rbp)
	jmp	.L410
.L411:
	movq	$10, -704(%rbp)
	jmp	.L410
.L404:
	movl	$0, -716(%rbp)
	movq	$12, -704(%rbp)
	jmp	.L410
.L409:
	movq	-712(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-768(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -712(%rbp)
	movq	$34, -704(%rbp)
	jmp	.L410
.L408:
	movl	-740(%rbp), %eax
	subl	$50, %eax
	movslq	%eax, %rcx
	movq	-712(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-712(%rbp), %rdx
	leaq	-688(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-712(%rbp), %rax
	leaq	-688(%rbp), %rdx
	leaq	52(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	$1, -704(%rbp)
	jmp	.L410
.L399:
	movq	-712(%rbp), %rdx
	leaq	-744(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-712(%rbp), %rdx
	leaq	-748(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-712(%rbp), %rdx
	leaq	-752(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	leaq	-688(%rbp), %rdx
	movl	-716(%rbp), %eax
	cltq
	addq	$28, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	8(%rax), %rdi
	movq	-712(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	leaq	-688(%rbp), %rdx
	movl	-716(%rbp), %eax
	cltq
	addq	$32, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	12(%rax), %rdi
	movq	-712(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	leaq	-688(%rbp), %rdx
	movl	-716(%rbp), %eax
	cltq
	addq	$40, %rax
	salq	$2, %rax
	leaq	(%rdx,%rax), %rdi
	movq	-712(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	leaq	-688(%rbp), %rcx
	movl	-716(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rcx, %rax
	leaq	4(%rax), %rdi
	movq	-712(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$50, %esi
	call	fwrite@PLT
	leaq	-688(%rbp), %rcx
	movl	-716(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	leaq	0(,%rax,4), %rdx
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rcx, %rax
	leaq	14(%rax), %rdi
	movq	-712(%rbp), %rax
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$50, %esi
	call	fwrite@PLT
	addl	$1, -716(%rbp)
	movq	$26, -704(%rbp)
	jmp	.L410
.L389:
	movl	-632(%rbp), %eax
	cmpl	%eax, -716(%rbp)
	jge	.L413
	movq	$32, -704(%rbp)
	jmp	.L410
.L413:
	movq	$22, -704(%rbp)
	jmp	.L410
.L396:
	cmpl	$4, -716(%rbp)
	jg	.L415
	movq	$21, -704(%rbp)
	jmp	.L410
.L415:
	movq	$31, -704(%rbp)
	jmp	.L410
.L401:
	movl	-632(%rbp), %eax
	cmpl	$5, %eax
	jle	.L417
	movq	$30, -704(%rbp)
	jmp	.L410
.L417:
	movq	$8, -704(%rbp)
	jmp	.L410
.L393:
	movl	-728(%rbp), %eax
	movl	-716(%rbp), %edx
	movslq	%edx, %rdx
	addq	$28, %rdx
	movl	%eax, -680(%rbp,%rdx,4)
	movl	-724(%rbp), %eax
	movl	-716(%rbp), %edx
	movslq	%edx, %rdx
	addq	$32, %rdx
	movl	%eax, -676(%rbp,%rdx,4)
	movl	-720(%rbp), %eax
	movl	-716(%rbp), %edx
	movslq	%edx, %rdx
	addq	$40, %rdx
	movl	%eax, -688(%rbp,%rdx,4)
	addl	$1, -716(%rbp)
	movq	$36, -704(%rbp)
	jmp	.L410
.L406:
	movl	-716(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-688(%rbp), %rsi
	movl	-716(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	addq	%rcx, %rax
	addq	%rax, %rax
	addq	$176, %rax
	addq	%rsi, %rax
	addq	$4, %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-688(%rbp), %rsi
	movl	-716(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	leaq	0(,%rax,4), %rcx
	addq	%rcx, %rax
	addq	%rax, %rax
	addq	$416, %rax
	addq	%rsi, %rax
	addq	$14, %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	addl	$1, -716(%rbp)
	movq	$12, -704(%rbp)
	jmp	.L410
.L391:
	cmpq	$0, -712(%rbp)
	jne	.L419
	movq	$7, -704(%rbp)
	jmp	.L410
.L419:
	movq	$4, -704(%rbp)
	jmp	.L410
.L398:
	movl	$0, -716(%rbp)
	movq	$26, -704(%rbp)
	jmp	.L410
.L392:
	movq	$25, -704(%rbp)
	jmp	.L410
.L403:
	movq	-712(%rbp), %rdx
	leaq	-688(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$50, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-712(%rbp), %rax
	leaq	-688(%rbp), %rdx
	leaq	52(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movq	-712(%rbp), %rax
	leaq	-688(%rbp), %rdx
	leaq	56(%rdx), %rdi
	movq	%rax, %rcx
	movl	$1, %edx
	movl	$4, %esi
	call	fwrite@PLT
	movl	-752(%rbp), %edx
	movl	-748(%rbp), %esi
	movl	-744(%rbp), %eax
	leaq	-720(%rbp), %r8
	leaq	-724(%rbp), %rdi
	leaq	-728(%rbp), %rcx
	movq	%r8, %r9
	movq	%rdi, %r8
	movl	%eax, %edi
	call	devolucao
	movl	$0, -716(%rbp)
	movq	$36, -704(%rbp)
	jmp	.L410
.L405:
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$31, -704(%rbp)
	jmp	.L410
.L390:
	cmpq	$0, -712(%rbp)
	jne	.L422
	movq	$14, -704(%rbp)
	jmp	.L410
.L422:
	movq	$3, -704(%rbp)
	jmp	.L410
.L427:
	nop
.L410:
	jmp	.L424
.L428:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L426
	call	__stack_chk_fail@PLT
.L426:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	novo_emprestimo, .-novo_emprestimo
	.section	.rodata
	.align 8
.LC52:
	.string	"Digite uma senha de pelo menos 4 algarismos"
.LC53:
	.string	"Senha numerica: "
.LC54:
	.string	"Digite a senha novamente: "
	.align 8
.LC55:
	.string	"Ja existe um cadastro nesse nome"
.LC56:
	.string	"Data de nascimento:"
.LC57:
	.string	"Endereco:"
.LC58:
	.string	"Rua: "
.LC59:
	.string	"Numero: "
.LC60:
	.string	"Complemento: "
.LC61:
	.string	"Bairro: "
.LC62:
	.string	"Cidade: "
.LC63:
	.string	"Estado: "
	.align 8
.LC64:
	.string	"Senha numerica [pelo menos 4 algarismos]: "
	.text
	.globl	cadastro
	.type	cadastro, @function
cadastro:
.LFB15:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -72(%rbp)
.L457:
	cmpq	$22, -72(%rbp)
	ja	.L460
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L432(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L432(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L432:
	.long	.L460-.L432
	.long	.L446-.L432
	.long	.L445-.L432
	.long	.L444-.L432
	.long	.L443-.L432
	.long	.L460-.L432
	.long	.L442-.L432
	.long	.L441-.L432
	.long	.L440-.L432
	.long	.L439-.L432
	.long	.L438-.L432
	.long	.L437-.L432
	.long	.L436-.L432
	.long	.L435-.L432
	.long	.L460-.L432
	.long	.L460-.L432
	.long	.L460-.L432
	.long	.L460-.L432
	.long	.L434-.L432
	.long	.L460-.L432
	.long	.L460-.L432
	.long	.L433-.L432
	.long	.L431-.L432
	.text
.L434:
	cmpl	$0, -80(%rbp)
	je	.L447
	movq	$7, -72(%rbp)
	jmp	.L449
.L447:
	movq	$22, -72(%rbp)
	jmp	.L449
.L443:
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-120(%rbp), %rcx
	leaq	-64(%rbp), %rdx
	movl	-108(%rbp), %esi
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	verifica_cad
	movl	%eax, -76(%rbp)
	movl	-76(%rbp), %eax
	movl	%eax, -80(%rbp)
	movq	$18, -72(%rbp)
	jmp	.L449
.L436:
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC53(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -84(%rbp)
	movq	$8, -72(%rbp)
	jmp	.L449
.L440:
	cmpl	$999, -84(%rbp)
	jg	.L450
	movq	$12, -72(%rbp)
	jmp	.L449
.L450:
	movq	$3, -72(%rbp)
	jmp	.L449
.L446:
	movl	$1, %eax
	jmp	.L458
.L444:
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movq	-104(%rbp), %rdx
	movl	%eax, 56(%rdx)
	movq	$11, -72(%rbp)
	jmp	.L449
.L433:
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -72(%rbp)
	jmp	.L449
.L437:
	movq	-104(%rbp), %rax
	movl	56(%rax), %eax
	cmpl	%eax, -84(%rbp)
	jne	.L453
	movq	$9, -72(%rbp)
	jmp	.L449
.L453:
	movq	$13, -72(%rbp)
	jmp	.L449
.L439:
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movq	-104(%rbp), %rdx
	movl	%eax, 60(%rdx)
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movq	-104(%rbp), %rdx
	movl	%eax, 64(%rdx)
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movq	-104(%rbp), %rdx
	movl	%eax, 68(%rdx)
	leaq	.LC57(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	.LC58(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-104(%rbp), %rax
	leaq	72(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movq	-104(%rbp), %rdx
	movl	%eax, 112(%rdx)
	leaq	.LC60(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-104(%rbp), %rax
	leaq	116(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	leaq	.LC61(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-104(%rbp), %rax
	leaq	136(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	leaq	.LC62(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-104(%rbp), %rax
	leaq	176(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	leaq	.LC63(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	-104(%rbp), %rax
	leaq	206(%rax), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	movq	$2, -72(%rbp)
	jmp	.L449
.L435:
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -84(%rbp)
	movq	$8, -72(%rbp)
	jmp	.L449
.L442:
	movq	$4, -72(%rbp)
	jmp	.L449
.L431:
	movq	-104(%rbp), %rax
	leaq	-64(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movl	-108(%rbp), %eax
	leal	100000(%rax), %edx
	movq	-104(%rbp), %rax
	movl	%edx, 52(%rax)
	movq	-104(%rbp), %rax
	movl	52(%rax), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -72(%rbp)
	jmp	.L449
.L438:
	movl	$-1, %eax
	jmp	.L458
.L441:
	cmpl	$-1, -80(%rbp)
	jne	.L455
	movq	$10, -72(%rbp)
	jmp	.L449
.L455:
	movq	$21, -72(%rbp)
	jmp	.L449
.L445:
	movl	$0, %eax
	jmp	.L458
.L460:
	nop
.L449:
	jmp	.L457
.L458:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L459
	call	__stack_chk_fail@PLT
.L459:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	cadastro, .-cadastro
	.section	.rodata
	.align 8
.LC65:
	.string	"Confirme sua senha (digite 0 para sair): "
.LC66:
	.string	"Senha incorreta"
	.text
	.globl	confirma_senha
	.type	confirma_senha, @function
confirma_senha:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movl	%ecx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$34, -80(%rbp)
.L516:
	cmpq	$41, -80(%rbp)
	ja	.L519
	movq	-80(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L464(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L464(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L464:
	.long	.L495-.L464
	.long	.L494-.L464
	.long	.L493-.L464
	.long	.L492-.L464
	.long	.L491-.L464
	.long	.L519-.L464
	.long	.L490-.L464
	.long	.L489-.L464
	.long	.L488-.L464
	.long	.L519-.L464
	.long	.L487-.L464
	.long	.L486-.L464
	.long	.L485-.L464
	.long	.L519-.L464
	.long	.L484-.L464
	.long	.L483-.L464
	.long	.L482-.L464
	.long	.L481-.L464
	.long	.L519-.L464
	.long	.L480-.L464
	.long	.L479-.L464
	.long	.L478-.L464
	.long	.L477-.L464
	.long	.L476-.L464
	.long	.L475-.L464
	.long	.L474-.L464
	.long	.L473-.L464
	.long	.L519-.L464
	.long	.L519-.L464
	.long	.L472-.L464
	.long	.L471-.L464
	.long	.L519-.L464
	.long	.L519-.L464
	.long	.L470-.L464
	.long	.L469-.L464
	.long	.L468-.L464
	.long	.L467-.L464
	.long	.L519-.L464
	.long	.L466-.L464
	.long	.L519-.L464
	.long	.L465-.L464
	.long	.L463-.L464
	.text
.L474:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$3, -80(%rbp)
	jmp	.L496
.L491:
	cmpq	$0, -96(%rbp)
	jne	.L497
	movq	$22, -80(%rbp)
	jmp	.L496
.L497:
	movq	$7, -80(%rbp)
	jmp	.L496
.L471:
	movq	-96(%rbp), %rax
	movl	$1, %edx
	movl	$174, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-96(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -88(%rbp)
	movq	$33, -80(%rbp)
	jmp	.L496
.L484:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$16, -80(%rbp)
	jmp	.L496
.L483:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$36, -80(%rbp)
	jmp	.L496
.L485:
	movl	$0, %eax
	jmp	.L517
.L488:
	cmpl	$0, -100(%rbp)
	jne	.L500
	movq	$19, -80(%rbp)
	jmp	.L496
.L500:
	movq	$30, -80(%rbp)
	jmp	.L496
.L494:
	movl	$0, %eax
	jmp	.L517
.L476:
	cmpl	$0, -104(%rbp)
	jne	.L502
	movq	$41, -80(%rbp)
	jmp	.L496
.L502:
	movq	$35, -80(%rbp)
	jmp	.L496
.L492:
	movl	$0, %eax
	jmp	.L517
.L482:
	movl	$1, %eax
	jmp	.L517
.L475:
	movl	-108(%rbp), %eax
	cmpl	%eax, -104(%rbp)
	jne	.L504
	movq	$14, -80(%rbp)
	jmp	.L496
.L504:
	movq	$23, -80(%rbp)
	jmp	.L496
.L478:
	cmpl	$2, -120(%rbp)
	jne	.L506
	movq	$17, -80(%rbp)
	jmp	.L496
.L506:
	movq	$25, -80(%rbp)
	jmp	.L496
.L467:
	movl	$0, %eax
	jmp	.L517
.L473:
	cmpl	$0, -104(%rbp)
	jne	.L508
	movq	$15, -80(%rbp)
	jmp	.L496
.L508:
	movq	$29, -80(%rbp)
	jmp	.L496
.L486:
	movq	-96(%rbp), %rax
	movl	$1, %edx
	movq	$-1, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$17, -80(%rbp)
	jmp	.L496
.L480:
	movq	-96(%rbp), %rax
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-96(%rbp), %rdx
	leaq	-108(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	$6, -80(%rbp)
	jmp	.L496
.L481:
	movq	-96(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rcx
	movl	$50, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-136(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -100(%rbp)
	movq	$8, -80(%rbp)
	jmp	.L496
.L465:
	movl	-108(%rbp), %eax
	cmpl	%eax, -104(%rbp)
	jne	.L510
	movq	$20, -80(%rbp)
	jmp	.L496
.L510:
	movq	$26, -80(%rbp)
	jmp	.L496
.L490:
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -104(%rbp)
	movq	$40, -80(%rbp)
	jmp	.L496
.L466:
	movl	$1, %eax
	jmp	.L517
.L469:
	movq	$10, -80(%rbp)
	jmp	.L496
.L477:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -80(%rbp)
	jmp	.L496
.L470:
	cmpq	$0, -88(%rbp)
	jne	.L512
	movq	$25, -80(%rbp)
	jmp	.L496
.L512:
	movq	$11, -80(%rbp)
	jmp	.L496
.L463:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$1, -80(%rbp)
	jmp	.L496
.L487:
	movq	-128(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$4, -80(%rbp)
	jmp	.L496
.L495:
	movl	-116(%rbp), %eax
	cltq
	addq	$4, %rax
	movq	%rax, %rcx
	movq	-96(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-96(%rbp), %rdx
	leaq	-108(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	$2, -80(%rbp)
	jmp	.L496
.L489:
	cmpl	$1, -120(%rbp)
	jne	.L514
	movq	$0, -80(%rbp)
	jmp	.L496
.L514:
	movq	$21, -80(%rbp)
	jmp	.L496
.L468:
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -80(%rbp)
	jmp	.L496
.L472:
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -80(%rbp)
	jmp	.L496
.L493:
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-64(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -104(%rbp)
	movq	$24, -80(%rbp)
	jmp	.L496
.L479:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$38, -80(%rbp)
	jmp	.L496
.L519:
	nop
.L496:
	jmp	.L516
.L517:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L518
	call	__stack_chk_fail@PLT
.L518:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	confirma_senha, .-confirma_senha
	.globl	devolucao
	.type	devolucao, @function
devolucao:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movq	%rcx, -56(%rbp)
	movq	%r8, -64(%rbp)
	movq	%r9, -72(%rbp)
	movq	$15, -8(%rbp)
.L544:
	cmpq	$15, -8(%rbp)
	ja	.L545
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L523(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L523(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L523:
	.long	.L545-.L523
	.long	.L533-.L523
	.long	.L532-.L523
	.long	.L545-.L523
	.long	.L531-.L523
	.long	.L530-.L523
	.long	.L545-.L523
	.long	.L546-.L523
	.long	.L545-.L523
	.long	.L528-.L523
	.long	.L545-.L523
	.long	.L527-.L523
	.long	.L526-.L523
	.long	.L525-.L523
	.long	.L524-.L523
	.long	.L522-.L523
	.text
.L531:
	subl	$1, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L534
.L524:
	cmpl	$12, -40(%rbp)
	jne	.L535
	movq	$1, -8(%rbp)
	jmp	.L534
.L535:
	movq	$12, -8(%rbp)
	jmp	.L534
.L522:
	movl	$31, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L534
.L526:
	movq	-72(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, (%rax)
	movl	-40(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -40(%rbp)
	movq	-64(%rbp), %rax
	movl	-16(%rbp), %edx
	movl	%edx, (%rax)
	movl	-36(%rbp), %eax
	addl	$14, %eax
	subl	-20(%rbp), %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	%edx, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L534
.L533:
	movl	-44(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -44(%rbp)
	movq	-72(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	-64(%rbp), %rax
	movl	$1, (%rax)
	movl	-36(%rbp), %eax
	addl	$14, %eax
	subl	-20(%rbp), %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	%edx, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L534
.L527:
	subl	$3, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L534
.L528:
	movq	$13, -8(%rbp)
	jmp	.L534
.L525:
	movl	-36(%rbp), %eax
	addl	$13, %eax
	cmpl	%eax, -20(%rbp)
	jg	.L537
	movq	$14, -8(%rbp)
	jmp	.L534
.L537:
	movq	$2, -8(%rbp)
	jmp	.L534
.L530:
	movl	-40(%rbp), %eax
	cmpl	$11, %eax
	seta	%dl
	testb	%dl, %dl
	jne	.L539
	movl	$1, %edx
	movl	%eax, %ecx
	salq	%cl, %rdx
	movq	%rdx, %rax
	movq	%rax, %rdx
	andl	$2640, %edx
	testq	%rdx, %rdx
	setne	%dl
	testb	%dl, %dl
	jne	.L540
	andl	$4, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	jne	.L541
	jmp	.L539
.L540:
	movq	$4, -8(%rbp)
	jmp	.L542
.L541:
	movq	$11, -8(%rbp)
	jmp	.L542
.L539:
	movq	$9, -8(%rbp)
	nop
.L542:
	jmp	.L534
.L532:
	movq	-72(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, (%rax)
	movq	-64(%rbp), %rax
	movl	-40(%rbp), %edx
	movl	%edx, (%rax)
	movl	-36(%rbp), %eax
	leal	14(%rax), %edx
	movq	-56(%rbp), %rax
	movl	%edx, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L534
.L545:
	nop
.L534:
	jmp	.L544
.L546:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	devolucao, .-devolucao
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
