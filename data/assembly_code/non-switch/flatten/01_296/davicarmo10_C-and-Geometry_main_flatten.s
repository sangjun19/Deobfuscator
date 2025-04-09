	.file	"davicarmo10_C-and-Geometry_main_flatten.c"
	.text
	.globl	_TIG_IZ_OAzv_argv
	.bss
	.align 8
	.type	_TIG_IZ_OAzv_argv, @object
	.size	_TIG_IZ_OAzv_argv, 8
_TIG_IZ_OAzv_argv:
	.zero	8
	.globl	_TIG_IZ_OAzv_envp
	.align 8
	.type	_TIG_IZ_OAzv_envp, @object
	.size	_TIG_IZ_OAzv_envp, 8
_TIG_IZ_OAzv_envp:
	.zero	8
	.globl	_TIG_IZ_OAzv_argc
	.align 4
	.type	_TIG_IZ_OAzv_argc, @object
	.size	_TIG_IZ_OAzv_argc, 4
_TIG_IZ_OAzv_argc:
	.zero	4
	.text
	.globl	determinant
	.type	determinant, @function
determinant:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$896, %rsp
	movq	%rdi, -888(%rbp)
	movl	%esi, -892(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -832(%rbp)
.L41:
	cmpq	$33, -832(%rbp)
	ja	.L44
	movq	-832(%rbp), %rax
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
	.long	.L24-.L4
	.long	.L44-.L4
	.long	.L44-.L4
	.long	.L23-.L4
	.long	.L44-.L4
	.long	.L44-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L44-.L4
	.long	.L44-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L44-.L4
	.long	.L12-.L4
	.long	.L44-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L44-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L44-.L4
	.long	.L44-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L44-.L4
	.long	.L5-.L4
	.long	.L44-.L4
	.long	.L3-.L4
	.text
.L8:
	movl	-872(%rbp), %eax
	cmpl	-892(%rbp), %eax
	jge	.L25
	movq	$13, -832(%rbp)
	jmp	.L27
.L25:
	movq	$15, -832(%rbp)
	jmp	.L27
.L16:
	movl	-856(%rbp), %eax
	cmpl	-892(%rbp), %eax
	jge	.L28
	movq	$21, -832(%rbp)
	jmp	.L27
.L28:
	movq	$8, -832(%rbp)
	jmp	.L27
.L15:
	movsd	-848(%rbp), %xmm0
	jmp	.L42
.L5:
	movq	-888(%rbp), %rax
	movsd	(%rax), %xmm1
	movq	-888(%rbp), %rax
	addq	$80, %rax
	movsd	8(%rax), %xmm0
	mulsd	%xmm1, %xmm0
	movq	-888(%rbp), %rax
	movsd	8(%rax), %xmm2
	movq	-888(%rbp), %rax
	addq	$80, %rax
	movsd	(%rax), %xmm1
	mulsd	%xmm2, %xmm1
	subsd	%xmm1, %xmm0
	jmp	.L42
.L18:
	movl	$0, -860(%rbp)
	movl	$0, -856(%rbp)
	movq	$14, -832(%rbp)
	jmp	.L27
.L20:
	addl	$1, -868(%rbp)
	addl	$1, -864(%rbp)
	movq	$19, -832(%rbp)
	jmp	.L27
.L23:
	movl	$0, -872(%rbp)
	movq	$25, -832(%rbp)
	jmp	.L27
.L14:
	cmpl	$1, -892(%rbp)
	jne	.L31
	movq	$0, -832(%rbp)
	jmp	.L27
.L31:
	movq	$29, -832(%rbp)
	jmp	.L27
.L9:
	movl	-872(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L33
	movq	$7, -832(%rbp)
	jmp	.L27
.L33:
	movq	$22, -832(%rbp)
	jmp	.L27
.L11:
	movl	-856(%rbp), %eax
	cmpl	-872(%rbp), %eax
	jne	.L35
	movq	$11, -832(%rbp)
	jmp	.L27
.L35:
	movq	$17, -832(%rbp)
	jmp	.L27
.L19:
	addl	$1, -856(%rbp)
	movq	$14, -832(%rbp)
	jmp	.L27
.L17:
	movl	$0, -868(%rbp)
	movl	$1, -864(%rbp)
	movq	$19, -832(%rbp)
	jmp	.L27
.L12:
	movl	-864(%rbp), %eax
	cmpl	-892(%rbp), %eax
	jge	.L37
	movq	$12, -832(%rbp)
	jmp	.L27
.L37:
	movq	$33, -832(%rbp)
	jmp	.L27
.L13:
	movl	-864(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-888(%rbp), %rax
	addq	%rax, %rdx
	movl	-856(%rbp), %eax
	cltq
	movsd	(%rdx,%rax,8), %xmm0
	movl	-860(%rbp), %eax
	movslq	%eax, %rcx
	movl	-868(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movsd	%xmm0, -816(%rbp,%rax,8)
	addl	$1, -860(%rbp)
	movq	$11, -832(%rbp)
	jmp	.L27
.L22:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -848(%rbp)
	movq	$16, -832(%rbp)
	jmp	.L27
.L10:
	movl	$-1, -852(%rbp)
	movq	$28, -832(%rbp)
	jmp	.L27
.L7:
	pxor	%xmm1, %xmm1
	cvtsi2sdl	-852(%rbp), %xmm1
	movq	-888(%rbp), %rax
	movl	-872(%rbp), %edx
	movslq	%edx, %rdx
	movsd	(%rax,%rdx,8), %xmm0
	mulsd	%xmm1, %xmm0
	mulsd	-840(%rbp), %xmm0
	movsd	-848(%rbp), %xmm1
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -848(%rbp)
	addl	$1, -872(%rbp)
	movq	$25, -832(%rbp)
	jmp	.L27
.L3:
	movl	-892(%rbp), %eax
	leal	-1(%rax), %edx
	leaq	-816(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	determinant
	movq	%xmm0, %rax
	movq	%rax, -824(%rbp)
	movsd	-824(%rbp), %xmm0
	movsd	%xmm0, -840(%rbp)
	movq	$24, -832(%rbp)
	jmp	.L27
.L24:
	movq	-888(%rbp), %rax
	movsd	(%rax), %xmm0
	jmp	.L42
.L21:
	movl	$1, -852(%rbp)
	movq	$28, -832(%rbp)
	jmp	.L27
.L6:
	cmpl	$2, -892(%rbp)
	jne	.L39
	movq	$31, -832(%rbp)
	jmp	.L27
.L39:
	movq	$3, -832(%rbp)
	jmp	.L27
.L44:
	nop
.L27:
	jmp	.L41
.L42:
	movq	%xmm0, %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	call	__stack_chk_fail@PLT
.L43:
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	determinant, .-determinant
	.section	.rodata
	.align 8
.LC1:
	.string	"Erro: Entrada inv\303\241lida. Certifique-se de fornecer um n\303\272mero entre 1 e %d.\n"
.LC2:
	.string	"%8.2lf "
	.align 8
.LC3:
	.string	"Erro: Entrada inv\303\241lida. Certifique-se de fornecer um n\303\272mero real."
.LC4:
	.string	"Elemento [%d][%d]: "
.LC5:
	.string	"%lf"
	.align 8
.LC6:
	.string	"Digite os elementos da matriz, linha por linha:"
	.align 8
.LC7:
	.string	"\nDeterminante da matriz: %.2lf\n"
.LC8:
	.string	"\nMatriz inserida:"
	.align 8
.LC9:
	.string	"Digite a ordem da matriz (1 a %d): "
.LC10:
	.string	"%d"
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
	subq	$912, %rsp
	movl	%edi, -884(%rbp)
	movq	%rsi, -896(%rbp)
	movq	%rdx, -904(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_OAzv_envp(%rip)
	nop
.L46:
	movq	$0, _TIG_IZ_OAzv_argv(%rip)
	nop
.L47:
	movl	$0, _TIG_IZ_OAzv_argc(%rip)
	nop
	nop
.L48:
.L49:
#APP
# 134 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-OAzv--0
# 0 "" 2
#NO_APP
	movl	-884(%rbp), %eax
	movl	%eax, _TIG_IZ_OAzv_argc(%rip)
	movq	-896(%rbp), %rax
	movq	%rax, _TIG_IZ_OAzv_argv(%rip)
	movq	-904(%rbp), %rax
	movq	%rax, _TIG_IZ_OAzv_envp(%rip)
	nop
	movq	$16, -840(%rbp)
.L99:
	cmpq	$42, -840(%rbp)
	ja	.L102
	movq	-840(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L52(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L52(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L52:
	.long	.L102-.L52
	.long	.L102-.L52
	.long	.L80-.L52
	.long	.L102-.L52
	.long	.L79-.L52
	.long	.L78-.L52
	.long	.L77-.L52
	.long	.L76-.L52
	.long	.L75-.L52
	.long	.L74-.L52
	.long	.L102-.L52
	.long	.L73-.L52
	.long	.L102-.L52
	.long	.L72-.L52
	.long	.L71-.L52
	.long	.L102-.L52
	.long	.L70-.L52
	.long	.L69-.L52
	.long	.L102-.L52
	.long	.L102-.L52
	.long	.L68-.L52
	.long	.L102-.L52
	.long	.L67-.L52
	.long	.L66-.L52
	.long	.L65-.L52
	.long	.L102-.L52
	.long	.L102-.L52
	.long	.L102-.L52
	.long	.L64-.L52
	.long	.L63-.L52
	.long	.L62-.L52
	.long	.L102-.L52
	.long	.L61-.L52
	.long	.L60-.L52
	.long	.L59-.L52
	.long	.L58-.L52
	.long	.L57-.L52
	.long	.L102-.L52
	.long	.L56-.L52
	.long	.L55-.L52
	.long	.L54-.L52
	.long	.L53-.L52
	.long	.L51-.L52
	.text
.L79:
	movl	-868(%rbp), %eax
	cmpl	%eax, -860(%rbp)
	jge	.L81
	movq	$34, -840(%rbp)
	jmp	.L83
.L81:
	movq	$7, -840(%rbp)
	jmp	.L83
.L62:
	cmpl	$1, -864(%rbp)
	je	.L84
	movq	$13, -840(%rbp)
	jmp	.L83
.L84:
	movq	$29, -840(%rbp)
	jmp	.L83
.L71:
	addl	$1, -856(%rbp)
	movq	$17, -840(%rbp)
	jmp	.L83
.L75:
	movl	$10, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$42, -840(%rbp)
	jmp	.L83
.L66:
	addl	$1, -860(%rbp)
	movq	$4, -840(%rbp)
	jmp	.L83
.L70:
	movq	$2, -840(%rbp)
	jmp	.L83
.L65:
	movl	$0, -844(%rbp)
	movq	$20, -840(%rbp)
	jmp	.L83
.L57:
	movl	-844(%rbp), %eax
	movslq	%eax, %rcx
	movl	-848(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movq	-816(%rbp,%rax,8), %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addl	$1, -844(%rbp)
	movq	$20, -840(%rbp)
	jmp	.L83
.L73:
	movl	$0, %eax
	jmp	.L100
.L74:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -848(%rbp)
	movq	$40, -840(%rbp)
	jmp	.L83
.L72:
	movl	$10, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$38, -840(%rbp)
	jmp	.L83
.L61:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -840(%rbp)
	jmp	.L83
.L69:
	movl	-868(%rbp), %eax
	cmpl	%eax, -856(%rbp)
	jge	.L87
	movq	$6, -840(%rbp)
	jmp	.L83
.L87:
	movq	$23, -840(%rbp)
	jmp	.L83
.L54:
	movl	-868(%rbp), %eax
	cmpl	%eax, -848(%rbp)
	jge	.L89
	movq	$24, -840(%rbp)
	jmp	.L83
.L89:
	movq	$41, -840(%rbp)
	jmp	.L83
.L77:
	movl	-856(%rbp), %eax
	leal	1(%rax), %edx
	movl	-860(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-816(%rbp), %rcx
	movl	-856(%rbp), %eax
	movslq	%eax, %rsi
	movl	-860(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rsi, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -852(%rbp)
	movq	$33, -840(%rbp)
	jmp	.L83
.L56:
	movl	$1, %eax
	jmp	.L100
.L59:
	movl	$0, -856(%rbp)
	movq	$17, -840(%rbp)
	jmp	.L83
.L67:
	movl	$1, %eax
	jmp	.L100
.L64:
	movl	$1, %eax
	jmp	.L100
.L78:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -860(%rbp)
	movq	$4, -840(%rbp)
	jmp	.L83
.L60:
	cmpl	$1, -852(%rbp)
	je	.L91
	movq	$32, -840(%rbp)
	jmp	.L83
.L91:
	movq	$14, -840(%rbp)
	jmp	.L83
.L53:
	movl	-868(%rbp), %edx
	leaq	-816(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	determinant
	movq	%xmm0, %rax
	movq	%rax, -832(%rbp)
	movsd	-832(%rbp), %xmm0
	movsd	%xmm0, -824(%rbp)
	movq	-824(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$11, -840(%rbp)
	jmp	.L83
.L51:
	movl	$1, %eax
	jmp	.L100
.L55:
	movl	$10, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$28, -840(%rbp)
	jmp	.L83
.L76:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -848(%rbp)
	movq	$40, -840(%rbp)
	jmp	.L83
.L58:
	movl	-868(%rbp), %eax
	cmpl	$10, %eax
	jle	.L93
	movq	$8, -840(%rbp)
	jmp	.L83
.L93:
	movq	$5, -840(%rbp)
	jmp	.L83
.L63:
	movl	-868(%rbp), %eax
	testl	%eax, %eax
	jg	.L95
	movq	$39, -840(%rbp)
	jmp	.L83
.L95:
	movq	$35, -840(%rbp)
	jmp	.L83
.L80:
	movl	$10, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-868(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -864(%rbp)
	movq	$30, -840(%rbp)
	jmp	.L83
.L68:
	movl	-868(%rbp), %eax
	cmpl	%eax, -844(%rbp)
	jge	.L97
	movq	$36, -840(%rbp)
	jmp	.L83
.L97:
	movq	$9, -840(%rbp)
	jmp	.L83
.L102:
	nop
.L83:
	jmp	.L99
.L100:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L101
	call	__stack_chk_fail@PLT
.L101:
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
