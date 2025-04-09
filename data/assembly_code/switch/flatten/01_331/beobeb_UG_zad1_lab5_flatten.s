	.file	"beobeb_UG_zad1_lab5_flatten.c"
	.text
	.globl	_TIG_IZ_0yLD_envp
	.bss
	.align 8
	.type	_TIG_IZ_0yLD_envp, @object
	.size	_TIG_IZ_0yLD_envp, 8
_TIG_IZ_0yLD_envp:
	.zero	8
	.globl	_TIG_IZ_0yLD_argc
	.align 4
	.type	_TIG_IZ_0yLD_argc, @object
	.size	_TIG_IZ_0yLD_argc, 4
_TIG_IZ_0yLD_argc:
	.zero	4
	.globl	_TIG_IZ_0yLD_argv
	.align 8
	.type	_TIG_IZ_0yLD_argv, @object
	.size	_TIG_IZ_0yLD_argv, 8
_TIG_IZ_0yLD_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
	.text
	.globl	odejmowanie
	.type	odejmowanie, @function
odejmowanie:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movl	-20(%rbp), %eax
	subl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	odejmowanie, .-odejmowanie
	.globl	dzielenie
	.type	dzielenie, @function
dzielenie:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L14:
	cmpq	$0, -8(%rbp)
	je	.L10
	cmpq	$1, -8(%rbp)
	jne	.L16
	jmp	.L15
.L10:
	movl	-20(%rbp), %eax
	cltd
	idivl	-24(%rbp)
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L13
.L16:
	nop
.L13:
	jmp	.L14
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	dzielenie, .-dzielenie
	.globl	mnozenie
	.type	mnozenie, @function
mnozenie:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L22:
	cmpq	$0, -8(%rbp)
	je	.L18
	cmpq	$1, -8(%rbp)
	jne	.L24
	jmp	.L23
.L18:
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L21
.L24:
	nop
.L21:
	jmp	.L22
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	mnozenie, .-mnozenie
	.section	.rodata
.LC1:
	.string	"wynik dzielenia: "
.LC2:
	.string	"Podaj dwie liczby: "
.LC3:
	.string	"%d %d"
	.align 8
.LC4:
	.string	"Podaj rodzaj dzia\305\202ania: \n1-odejmowanie \n2-dodawanie \n3-mnozenie \n4-dzielenie"
.LC5:
	.string	"wynik mnozenia: "
.LC6:
	.string	"wynik dodawania: "
.LC7:
	.string	"wynik odejmowania: "
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_0yLD_envp(%rip)
	nop
.L26:
	movq	$0, _TIG_IZ_0yLD_argv(%rip)
	nop
.L27:
	movl	$0, _TIG_IZ_0yLD_argc(%rip)
	nop
	nop
.L28:
.L29:
#APP
# 83 "beobeb_UG_zad1_lab5.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0yLD--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_0yLD_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_0yLD_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_0yLD_envp(%rip)
	nop
	movq	$12, -16(%rbp)
.L49:
	cmpq	$15, -16(%rbp)
	ja	.L52
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L40-.L32
	.long	.L39-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L38-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L52-.L32
	.long	.L37-.L32
	.long	.L36-.L32
	.long	.L52-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L33-.L32
	.long	.L31-.L32
	.text
.L33:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	dzielenie
	movq	$9, -16(%rbp)
	jmp	.L41
.L31:
	movq	$9, -16(%rbp)
	jmp	.L41
.L35:
	movq	$13, -16(%rbp)
	jmp	.L41
.L39:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L42
	cmpl	$4, %eax
	jg	.L43
	cmpl	$3, %eax
	je	.L44
	cmpl	$3, %eax
	jg	.L43
	cmpl	$1, %eax
	je	.L45
	cmpl	$2, %eax
	je	.L46
	jmp	.L43
.L42:
	movq	$14, -16(%rbp)
	jmp	.L47
.L44:
	movq	$5, -16(%rbp)
	jmp	.L47
.L46:
	movq	$10, -16(%rbp)
	jmp	.L47
.L45:
	movq	$0, -16(%rbp)
	jmp	.L47
.L43:
	movq	$15, -16(%rbp)
	nop
.L47:
	jmp	.L41
.L37:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L50
	jmp	.L51
.L34:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -16(%rbp)
	jmp	.L41
.L38:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	mnozenie
	movq	$9, -16(%rbp)
	jmp	.L41
.L36:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	dodawanie
	movq	$9, -16(%rbp)
	jmp	.L41
.L40:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	odejmowanie
	movq	$9, -16(%rbp)
	jmp	.L41
.L52:
	nop
.L41:
	jmp	.L49
.L51:
	call	__stack_chk_fail@PLT
.L50:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	dodawanie
	.type	dodawanie, @function
dodawanie:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L58:
	cmpq	$0, -8(%rbp)
	je	.L54
	cmpq	$1, -8(%rbp)
	jne	.L60
	jmp	.L59
.L54:
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L57
.L60:
	nop
.L57:
	jmp	.L58
.L59:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	dodawanie, .-dodawanie
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
