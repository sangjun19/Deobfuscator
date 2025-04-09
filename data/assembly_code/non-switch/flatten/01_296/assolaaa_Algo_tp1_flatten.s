	.file	"assolaaa_Algo_tp1_flatten.c"
	.text
	.globl	_TIG_IZ_wFCO_argv
	.bss
	.align 8
	.type	_TIG_IZ_wFCO_argv, @object
	.size	_TIG_IZ_wFCO_argv, 8
_TIG_IZ_wFCO_argv:
	.zero	8
	.globl	_TIG_IZ_wFCO_envp
	.align 8
	.type	_TIG_IZ_wFCO_envp, @object
	.size	_TIG_IZ_wFCO_envp, 8
_TIG_IZ_wFCO_envp:
	.zero	8
	.globl	_TIG_IZ_wFCO_argc
	.align 4
	.type	_TIG_IZ_wFCO_argc, @object
	.size	_TIG_IZ_wFCO_argc, 4
_TIG_IZ_wFCO_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"\t\t\t  %c"
.LC1:
	.string	"A[%d][%d]= "
.LC2:
	.string	"\n%d"
.LC3:
	.string	"\tAvant: "
.LC4:
	.string	"\t%c"
.LC5:
	.string	"Exercice 02:"
.LC6:
	.string	"donne n :"
.LC7:
	.string	"%d"
.LC8:
	.string	"\ndonne la matrice :"
.LC9:
	.string	"Exercice 01:\n   Question 1.1:"
.LC10:
	.string	"\t%d"
.LC11:
	.string	"\n pair="
.LC12:
	.string	"\n impair="
.LC13:
	.string	"   Question 1.2:"
	.align 8
.LC14:
	.string	"\nla 1e diagonale=\tla 2e diagonale="
.LC15:
	.string	"\tApres: "
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
	subq	$3312, %rsp
	movl	%edi, -3284(%rbp)
	movq	%rsi, -3296(%rbp)
	movq	%rdx, -3304(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_wFCO_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_wFCO_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_wFCO_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wFCO--0
# 0 "" 2
#NO_APP
	movl	-3284(%rbp), %eax
	movl	%eax, _TIG_IZ_wFCO_argc(%rip)
	movq	-3296(%rbp), %rax
	movq	%rax, _TIG_IZ_wFCO_argv(%rip)
	movq	-3304(%rbp), %rax
	movq	%rax, _TIG_IZ_wFCO_envp(%rip)
	nop
	movq	$55, -3256(%rbp)
.L105:
	cmpq	$109, -3256(%rbp)
	ja	.L108
	movq	-3256(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L108-.L8
	.long	.L64-.L8
	.long	.L108-.L8
	.long	.L63-.L8
	.long	.L62-.L8
	.long	.L61-.L8
	.long	.L108-.L8
	.long	.L60-.L8
	.long	.L59-.L8
	.long	.L58-.L8
	.long	.L57-.L8
	.long	.L56-.L8
	.long	.L55-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L54-.L8
	.long	.L53-.L8
	.long	.L108-.L8
	.long	.L52-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L51-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L50-.L8
	.long	.L108-.L8
	.long	.L49-.L8
	.long	.L108-.L8
	.long	.L48-.L8
	.long	.L108-.L8
	.long	.L47-.L8
	.long	.L108-.L8
	.long	.L46-.L8
	.long	.L45-.L8
	.long	.L44-.L8
	.long	.L108-.L8
	.long	.L43-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L42-.L8
	.long	.L108-.L8
	.long	.L41-.L8
	.long	.L108-.L8
	.long	.L40-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L39-.L8
	.long	.L108-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L108-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L108-.L8
	.long	.L30-.L8
	.long	.L108-.L8
	.long	.L29-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L25-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L21-.L8
	.long	.L108-.L8
	.long	.L20-.L8
	.long	.L108-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L17-.L8
	.long	.L108-.L8
	.long	.L16-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L108-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L108-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L52:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3264(%rbp)
	jge	.L65
	movq	$30, -3256(%rbp)
	jmp	.L67
.L65:
	movq	$1, -3256(%rbp)
	jmp	.L67
.L24:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3264(%rbp)
	jge	.L68
	movq	$71, -3256(%rbp)
	jmp	.L67
.L68:
	movq	$96, -3256(%rbp)
	jmp	.L67
.L9:
	addl	$1, -3260(%rbp)
	movq	$61, -3256(%rbp)
	jmp	.L67
.L62:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3260(%rbp)
	jge	.L70
	movq	$35, -3256(%rbp)
	jmp	.L67
.L70:
	movq	$103, -3256(%rbp)
	jmp	.L67
.L48:
	movl	$0, -3260(%rbp)
	movq	$61, -3256(%rbp)
	jmp	.L67
.L13:
	addl	$1, -3264(%rbp)
	movq	$9, -3256(%rbp)
	jmp	.L67
.L14:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	-1616(%rbp,%rax,4), %edx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rsi
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	movl	%edx, -3216(%rbp,%rax,4)
	movq	$77, -3256(%rbp)
	jmp	.L67
.L11:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3264(%rbp)
	jge	.L72
	movq	$10, -3256(%rbp)
	jmp	.L67
.L72:
	movq	$38, -3256(%rbp)
	jmp	.L67
.L54:
	cmpl	$4, -3260(%rbp)
	jg	.L74
	movq	$73, -3256(%rbp)
	jmp	.L67
.L74:
	movq	$86, -3256(%rbp)
	jmp	.L67
.L22:
	movl	$10, %edi
	call	putchar@PLT
	addl	$2, -3264(%rbp)
	movq	$12, -3256(%rbp)
	jmp	.L67
.L35:
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rbp, %rax
	subq	$3248, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rbp, %rax
	addq	%rcx, %rax
	subq	$3248, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -3264(%rbp)
	subl	$1, -3260(%rbp)
	movq	$7, -3256(%rbp)
	jmp	.L67
.L55:
	cmpl	$4, -3264(%rbp)
	jg	.L76
	movq	$81, -3256(%rbp)
	jmp	.L67
.L76:
	movq	$58, -3256(%rbp)
	jmp	.L67
.L15:
	movl	-3260(%rbp), %edx
	movl	-3264(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-3216(%rbp), %rcx
	movl	-3260(%rbp), %eax
	movslq	%eax, %rsi
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -3260(%rbp)
	movq	$63, -3256(%rbp)
	jmp	.L67
.L59:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3260(%rbp)
	jge	.L78
	movq	$72, -3256(%rbp)
	jmp	.L67
.L78:
	movq	$5, -3256(%rbp)
	jmp	.L67
.L16:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -3264(%rbp)
	movq	$107, -3256(%rbp)
	jmp	.L67
.L41:
	movl	-3264(%rbp), %eax
	cmpl	-3260(%rbp), %eax
	je	.L80
	movq	$54, -3256(%rbp)
	jmp	.L67
.L80:
	movq	$108, -3256(%rbp)
	jmp	.L67
.L37:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	-3216(%rbp,%rax,4), %edx
	movl	-3260(%rbp), %eax
	movslq	%eax, %rsi
	movl	-3264(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	movl	%edx, -1616(%rbp,%rax,4)
	movl	-3264(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3260(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	-3216(%rbp,%rax,4), %edx
	movl	-3260(%rbp), %eax
	movslq	%eax, %rsi
	movl	-3264(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	movl	%edx, -3216(%rbp,%rax,4)
	movq	$108, -3256(%rbp)
	jmp	.L67
.L64:
	movl	$0, -3264(%rbp)
	movq	$9, -3256(%rbp)
	jmp	.L67
.L23:
	movl	$0, -3260(%rbp)
	movq	$22, -3256(%rbp)
	jmp	.L67
.L25:
	addl	$1, -3260(%rbp)
	movq	$4, -3256(%rbp)
	jmp	.L67
.L63:
	cmpl	$4, -3260(%rbp)
	jg	.L82
	movq	$51, -3256(%rbp)
	jmp	.L67
.L82:
	movq	$34, -3256(%rbp)
	jmp	.L67
.L53:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rbp, %rax
	addq	%rcx, %rax
	subq	$3248, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -3260(%rbp)
	movq	$22, -3256(%rbp)
	jmp	.L67
.L17:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -3264(%rbp)
	movq	$107, -3256(%rbp)
	jmp	.L67
.L7:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-3268(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -3264(%rbp)
	movq	$80, -3256(%rbp)
	jmp	.L67
.L44:
	movl	$0, -3260(%rbp)
	movq	$3, -3256(%rbp)
	jmp	.L67
.L34:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	movb	$49, -3248(%rbp)
	movb	$50, -3247(%rbp)
	movb	$51, -3246(%rbp)
	movb	$52, -3245(%rbp)
	movb	$53, -3244(%rbp)
	movb	$55, -3243(%rbp)
	movb	$97, -3242(%rbp)
	movb	$99, -3241(%rbp)
	movb	$56, -3240(%rbp)
	movb	$100, -3239(%rbp)
	movb	$99, -3238(%rbp)
	movb	$57, -3237(%rbp)
	movb	$52, -3236(%rbp)
	movb	$122, -3235(%rbp)
	movb	$56, -3234(%rbp)
	movb	$53, -3233(%rbp)
	movb	$54, -3232(%rbp)
	movb	$112, -3231(%rbp)
	movb	$110, -3230(%rbp)
	movb	$51, -3229(%rbp)
	movb	$50, -3228(%rbp)
	movb	$57, -3227(%rbp)
	movb	$116, -3226(%rbp)
	movb	$109, -3225(%rbp)
	movb	$107, -3224(%rbp)
	movl	$0, -3264(%rbp)
	movq	$47, -3256(%rbp)
	jmp	.L67
.L50:
	movl	$0, -3260(%rbp)
	movq	$4, -3256(%rbp)
	jmp	.L67
.L12:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	-3216(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -3260(%rbp)
	movq	$11, -3256(%rbp)
	jmp	.L67
.L56:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3260(%rbp)
	jge	.L84
	movq	$105, -3256(%rbp)
	jmp	.L67
.L84:
	movq	$94, -3256(%rbp)
	jmp	.L67
.L58:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3264(%rbp)
	jge	.L86
	movq	$26, -3256(%rbp)
	jmp	.L67
.L86:
	movq	$91, -3256(%rbp)
	jmp	.L67
.L30:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3260(%rbp)
	jge	.L88
	movq	$101, -3256(%rbp)
	jmp	.L67
.L88:
	movq	$88, -3256(%rbp)
	jmp	.L67
.L39:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rbp, %rax
	addq	%rcx, %rax
	subq	$3248, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -3260(%rbp)
	movq	$3, -3256(%rbp)
	jmp	.L67
.L10:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3264(%rbp)
	jge	.L90
	movq	$53, -3256(%rbp)
	jmp	.L67
.L90:
	movq	$60, -3256(%rbp)
	jmp	.L67
.L47:
	addl	$1, -3264(%rbp)
	movq	$18, -3256(%rbp)
	jmp	.L67
.L19:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -3264(%rbp)
	movq	$12, -3256(%rbp)
	jmp	.L67
.L36:
	movq	$57, -3256(%rbp)
	jmp	.L67
.L32:
	movl	$0, -3264(%rbp)
	movq	$18, -3256(%rbp)
	jmp	.L67
.L43:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L106
	jmp	.L107
.L31:
	movl	-3268(%rbp), %eax
	cmpl	%eax, -3260(%rbp)
	jge	.L93
	movq	$45, -3256(%rbp)
	jmp	.L67
.L93:
	movq	$32, -3256(%rbp)
	jmp	.L67
.L33:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -3264(%rbp)
	movq	$43, -3256(%rbp)
	jmp	.L67
.L46:
	movl	$10, %edi
	call	putchar@PLT
	addl	$2, -3264(%rbp)
	movq	$43, -3256(%rbp)
	jmp	.L67
.L28:
	movl	$0, -3260(%rbp)
	movq	$63, -3256(%rbp)
	jmp	.L67
.L51:
	cmpl	$4, -3260(%rbp)
	jg	.L95
	movq	$16, -3256(%rbp)
	jmp	.L67
.L95:
	movq	$82, -3256(%rbp)
	jmp	.L67
.L49:
	movl	$0, -3260(%rbp)
	movq	$15, -3256(%rbp)
	jmp	.L67
.L38:
	movl	$0, -3260(%rbp)
	movq	$11, -3256(%rbp)
	jmp	.L67
.L29:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -3264(%rbp)
	movl	$4, -3260(%rbp)
	movq	$7, -3256(%rbp)
	jmp	.L67
.L40:
	cmpl	$4, -3264(%rbp)
	jg	.L97
	movq	$28, -3256(%rbp)
	jmp	.L67
.L97:
	movq	$90, -3256(%rbp)
	jmp	.L67
.L26:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rbp, %rax
	addq	%rcx, %rax
	subq	$3248, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -3260(%rbp)
	movq	$15, -3256(%rbp)
	jmp	.L67
.L61:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -3264(%rbp)
	movq	$106, -3256(%rbp)
	jmp	.L67
.L18:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -3264(%rbp)
	movq	$106, -3256(%rbp)
	jmp	.L67
.L27:
	movl	-3260(%rbp), %eax
	movslq	%eax, %rcx
	movl	-3264(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	-3216(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -3260(%rbp)
	movq	$8, -3256(%rbp)
	jmp	.L67
.L57:
	movl	$0, -3260(%rbp)
	movq	$8, -3256(%rbp)
	jmp	.L67
.L60:
	cmpl	$4, -3264(%rbp)
	jg	.L99
	movq	$56, -3256(%rbp)
	jmp	.L67
.L99:
	movq	$109, -3256(%rbp)
	jmp	.L67
.L20:
	addl	$1, -3264(%rbp)
	movq	$80, -3256(%rbp)
	jmp	.L67
.L45:
	movl	-3264(%rbp), %eax
	cmpl	-3260(%rbp), %eax
	je	.L101
	movq	$102, -3256(%rbp)
	jmp	.L67
.L101:
	movq	$77, -3256(%rbp)
	jmp	.L67
.L42:
	cmpl	$4, -3264(%rbp)
	jg	.L103
	movq	$36, -3256(%rbp)
	jmp	.L67
.L103:
	movq	$65, -3256(%rbp)
	jmp	.L67
.L21:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -3264(%rbp)
	movq	$47, -3256(%rbp)
	jmp	.L67
.L108:
	nop
.L67:
	jmp	.L105
.L107:
	call	__stack_chk_fail@PLT
.L106:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
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
