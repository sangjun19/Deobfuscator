	.file	"shioimm_til_fsnav0_flatten.c"
	.text
	.globl	_TIG_IZ_YfHV_argv
	.bss
	.align 8
	.type	_TIG_IZ_YfHV_argv, @object
	.size	_TIG_IZ_YfHV_argv, 8
_TIG_IZ_YfHV_argv:
	.zero	8
	.globl	termcur
	.align 32
	.type	termcur, @object
	.size	termcur, 60
termcur:
	.zero	60
	.globl	_TIG_IZ_YfHV_argc
	.align 4
	.type	_TIG_IZ_YfHV_argc, @object
	.size	_TIG_IZ_YfHV_argc, 4
_TIG_IZ_YfHV_argc:
	.zero	4
	.globl	termsave
	.align 32
	.type	termsave, @object
	.size	termsave, 60
termsave:
	.zero	60
	.globl	_TIG_IZ_YfHV_envp
	.align 8
	.type	_TIG_IZ_YfHV_envp, @object
	.size	_TIG_IZ_YfHV_envp, 8
_TIG_IZ_YfHV_envp:
	.zero	8
	.text
	.globl	showcwd
	.type	showcwd, @function
showcwd:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -4120(%rbp)
.L7:
	cmpq	$2, -4120(%rbp)
	je	.L2
	cmpq	$2, -4120(%rbp)
	ja	.L11
	cmpq	$0, -4120(%rbp)
	je	.L4
	cmpq	$1, -4120(%rbp)
	jne	.L11
	jmp	.L10
.L4:
	movq	$2, -4120(%rbp)
	jmp	.L6
.L2:
	leaq	-4112(%rbp), %rax
	movl	$4097, %esi
	movq	%rax, %rdi
	call	getcwd@PLT
	leaq	-4112(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -4120(%rbp)
	jmp	.L6
.L11:
	nop
.L6:
	jmp	.L7
.L10:
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	showcwd, .-showcwd
	.section	.rodata
.LC0:
	.string	"chdir"
.LC1:
	.string	"directory? "
	.text
	.globl	godown
	.type	godown, @function
godown:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -4120(%rbp)
.L30:
	cmpq	$8, -4120(%rbp)
	ja	.L33
	movq	-4120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L15(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L15(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L15:
	.long	.L23-.L15
	.long	.L22-.L15
	.long	.L21-.L15
	.long	.L20-.L15
	.long	.L19-.L15
	.long	.L18-.L15
	.long	.L17-.L15
	.long	.L16-.L15
	.long	.L34-.L15
	.text
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$1, -4120(%rbp)
	jmp	.L24
.L22:
	leaq	termcur(%rip), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movq	$8, -4120(%rbp)
	jmp	.L24
.L20:
	cmpl	$0, -4132(%rbp)
	je	.L26
	movq	$4, -4120(%rbp)
	jmp	.L24
.L26:
	movq	$1, -4120(%rbp)
	jmp	.L24
.L17:
	movq	-4128(%rbp), %rax
	movb	$0, (%rax)
	movq	$7, -4120(%rbp)
	jmp	.L24
.L18:
	cmpq	$0, -4128(%rbp)
	je	.L28
	movq	$6, -4120(%rbp)
	jmp	.L24
.L28:
	movq	$7, -4120(%rbp)
	jmp	.L24
.L23:
	movq	$2, -4120(%rbp)
	jmp	.L24
.L16:
	leaq	-4112(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	movl	%eax, -4132(%rbp)
	movq	$3, -4120(%rbp)
	jmp	.L24
.L21:
	leaq	termsave(%rip), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$11, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	stdin(%rip), %rdx
	leaq	-4112(%rbp), %rax
	movl	$4097, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-4112(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -4128(%rbp)
	movq	$5, -4120(%rbp)
	jmp	.L24
.L33:
	nop
.L24:
	jmp	.L30
.L34:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L32
	call	__stack_chk_fail@PLT
.L32:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	godown, .-godown
	.section	.rodata
.LC2:
	.string	".."
	.text
	.globl	goup
	.type	goup, @function
goup:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L40:
	cmpq	$0, -8(%rbp)
	je	.L41
	cmpq	$1, -8(%rbp)
	jne	.L42
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	movq	$0, -8(%rbp)
	jmp	.L38
.L42:
	nop
.L38:
	jmp	.L40
.L41:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	goup, .-goup
	.globl	restoreterm
	.type	restoreterm, @function
restoreterm:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L48:
	cmpq	$0, -8(%rbp)
	je	.L49
	cmpq	$1, -8(%rbp)
	jne	.L50
	leaq	termsave(%rip), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movq	$0, -8(%rbp)
	jmp	.L46
.L50:
	nop
.L46:
	jmp	.L48
.L49:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	restoreterm, .-restoreterm
	.section	.rodata
.LC3:
	.string	"unknown command '%c'\n"
.LC4:
	.string	"atexit"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, termcur(%rip)
	movl	$0, 4+termcur(%rip)
	movl	$0, 8+termcur(%rip)
	movl	$0, 12+termcur(%rip)
	movb	$0, 16+termcur(%rip)
	movb	$0, 17+termcur(%rip)
	movb	$0, 18+termcur(%rip)
	movb	$0, 19+termcur(%rip)
	movb	$0, 20+termcur(%rip)
	movb	$0, 21+termcur(%rip)
	movb	$0, 22+termcur(%rip)
	movb	$0, 23+termcur(%rip)
	movb	$0, 24+termcur(%rip)
	movb	$0, 25+termcur(%rip)
	movb	$0, 26+termcur(%rip)
	movb	$0, 27+termcur(%rip)
	movb	$0, 28+termcur(%rip)
	movb	$0, 29+termcur(%rip)
	movb	$0, 30+termcur(%rip)
	movb	$0, 31+termcur(%rip)
	movb	$0, 32+termcur(%rip)
	movb	$0, 33+termcur(%rip)
	movb	$0, 34+termcur(%rip)
	movb	$0, 35+termcur(%rip)
	movb	$0, 36+termcur(%rip)
	movb	$0, 37+termcur(%rip)
	movb	$0, 38+termcur(%rip)
	movb	$0, 39+termcur(%rip)
	movb	$0, 40+termcur(%rip)
	movb	$0, 41+termcur(%rip)
	movb	$0, 42+termcur(%rip)
	movb	$0, 43+termcur(%rip)
	movb	$0, 44+termcur(%rip)
	movb	$0, 45+termcur(%rip)
	movb	$0, 46+termcur(%rip)
	movb	$0, 47+termcur(%rip)
	movb	$0, 48+termcur(%rip)
	movl	$0, 52+termcur(%rip)
	movl	$0, 56+termcur(%rip)
	nop
.L52:
	movl	$0, termsave(%rip)
	movl	$0, 4+termsave(%rip)
	movl	$0, 8+termsave(%rip)
	movl	$0, 12+termsave(%rip)
	movb	$0, 16+termsave(%rip)
	movb	$0, 17+termsave(%rip)
	movb	$0, 18+termsave(%rip)
	movb	$0, 19+termsave(%rip)
	movb	$0, 20+termsave(%rip)
	movb	$0, 21+termsave(%rip)
	movb	$0, 22+termsave(%rip)
	movb	$0, 23+termsave(%rip)
	movb	$0, 24+termsave(%rip)
	movb	$0, 25+termsave(%rip)
	movb	$0, 26+termsave(%rip)
	movb	$0, 27+termsave(%rip)
	movb	$0, 28+termsave(%rip)
	movb	$0, 29+termsave(%rip)
	movb	$0, 30+termsave(%rip)
	movb	$0, 31+termsave(%rip)
	movb	$0, 32+termsave(%rip)
	movb	$0, 33+termsave(%rip)
	movb	$0, 34+termsave(%rip)
	movb	$0, 35+termsave(%rip)
	movb	$0, 36+termsave(%rip)
	movb	$0, 37+termsave(%rip)
	movb	$0, 38+termsave(%rip)
	movb	$0, 39+termsave(%rip)
	movb	$0, 40+termsave(%rip)
	movb	$0, 41+termsave(%rip)
	movb	$0, 42+termsave(%rip)
	movb	$0, 43+termsave(%rip)
	movb	$0, 44+termsave(%rip)
	movb	$0, 45+termsave(%rip)
	movb	$0, 46+termsave(%rip)
	movb	$0, 47+termsave(%rip)
	movb	$0, 48+termsave(%rip)
	movl	$0, 52+termsave(%rip)
	movl	$0, 56+termsave(%rip)
	nop
.L53:
	movq	$0, _TIG_IZ_YfHV_envp(%rip)
	nop
.L54:
	movq	$0, _TIG_IZ_YfHV_argv(%rip)
	nop
.L55:
	movl	$0, _TIG_IZ_YfHV_argc(%rip)
	nop
	nop
.L56:
.L57:
#APP
# 165 "shioimm_til_fsnav0.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-YfHV--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_YfHV_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_YfHV_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_YfHV_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L83:
	cmpq	$21, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L72-.L60
	.long	.L84-.L60
	.long	.L84-.L60
	.long	.L71-.L60
	.long	.L84-.L60
	.long	.L70-.L60
	.long	.L84-.L60
	.long	.L69-.L60
	.long	.L68-.L60
	.long	.L84-.L60
	.long	.L84-.L60
	.long	.L84-.L60
	.long	.L67-.L60
	.long	.L66-.L60
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L63-.L60
	.long	.L62-.L60
	.long	.L84-.L60
	.long	.L84-.L60
	.long	.L61-.L60
	.long	.L59-.L60
	.text
.L65:
	movq	stderr(%rip), %rax
	movl	-16(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$13, -8(%rbp)
	jmp	.L73
.L64:
	movl	$0, %edi
	call	exit@PLT
.L67:
	cmpl	$117, -16(%rbp)
	je	.L74
	cmpl	$117, -16(%rbp)
	jg	.L75
	cmpl	$113, -16(%rbp)
	je	.L76
	cmpl	$113, -16(%rbp)
	jg	.L75
	cmpl	$108, -16(%rbp)
	je	.L77
	cmpl	$108, -16(%rbp)
	jg	.L75
	cmpl	$63, -16(%rbp)
	je	.L78
	cmpl	$100, -16(%rbp)
	je	.L79
	jmp	.L75
.L76:
	movq	$15, -8(%rbp)
	jmp	.L80
.L78:
	movq	$21, -8(%rbp)
	jmp	.L80
.L79:
	movq	$17, -8(%rbp)
	jmp	.L80
.L74:
	movq	$20, -8(%rbp)
	jmp	.L80
.L77:
	movq	$7, -8(%rbp)
	jmp	.L80
.L75:
	movq	$14, -8(%rbp)
	nop
.L80:
	jmp	.L73
.L68:
	cmpl	$0, -12(%rbp)
	je	.L81
	movq	$0, -8(%rbp)
	jmp	.L73
.L81:
	movq	$16, -8(%rbp)
	jmp	.L73
.L71:
	movq	$5, -8(%rbp)
	jmp	.L73
.L63:
	movl	12+termcur(%rip), %eax
	andl	$-11, %eax
	movl	%eax, 12+termcur(%rip)
	movb	$1, 23+termcur(%rip)
	movb	$0, 22+termcur(%rip)
	leaq	termcur(%rip), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movq	stdin(%rip), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	setbuf@PLT
	movq	$13, -8(%rbp)
	jmp	.L73
.L59:
	call	showcwd
	movq	$13, -8(%rbp)
	jmp	.L73
.L66:
	call	getchar@PLT
	movl	%eax, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L73
.L62:
	call	godown
	call	showcwd
	movq	$13, -8(%rbp)
	jmp	.L73
.L70:
	leaq	termsave(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	tcgetattr@PLT
	movq	termsave(%rip), %rax
	movq	8+termsave(%rip), %rdx
	movq	%rax, termcur(%rip)
	movq	%rdx, 8+termcur(%rip)
	movq	16+termsave(%rip), %rax
	movq	24+termsave(%rip), %rdx
	movq	%rax, 16+termcur(%rip)
	movq	%rdx, 24+termcur(%rip)
	movq	32+termsave(%rip), %rax
	movq	40+termsave(%rip), %rdx
	movq	%rax, 32+termcur(%rip)
	movq	%rdx, 40+termcur(%rip)
	movq	48+termsave(%rip), %rax
	movq	%rax, 48+termcur(%rip)
	movl	56+termsave(%rip), %eax
	movl	%eax, 56+termcur(%rip)
	leaq	restoreterm(%rip), %rax
	movq	%rax, %rdi
	call	atexit@PLT
	movl	%eax, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L73
.L72:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L69:
	call	showlist
	movq	$13, -8(%rbp)
	jmp	.L73
.L61:
	call	goup
	call	showcwd
	movq	$13, -8(%rbp)
	jmp	.L73
.L84:
	nop
.L73:
	jmp	.L83
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC5:
	.string	"."
.LC6:
	.string	"cannot open directory.\n"
.LC7:
	.string	"\nreaddir"
	.text
	.globl	showlist
	.type	showlist, @function
showlist:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	$15, -16(%rbp)
.L110:
	cmpq	$16, -16(%rbp)
	ja	.L111
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L88(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L88(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L88:
	.long	.L101-.L88
	.long	.L100-.L88
	.long	.L111-.L88
	.long	.L112-.L88
	.long	.L98-.L88
	.long	.L111-.L88
	.long	.L97-.L88
	.long	.L112-.L88
	.long	.L95-.L88
	.long	.L111-.L88
	.long	.L94-.L88
	.long	.L93-.L88
	.long	.L92-.L88
	.long	.L91-.L88
	.long	.L90-.L88
	.long	.L89-.L88
	.long	.L87-.L88
	.text
.L98:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L102
	movq	$1, -16(%rbp)
	jmp	.L104
.L102:
	movq	$11, -16(%rbp)
	jmp	.L104
.L90:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -32(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L104
.L89:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -40(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L104
.L92:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$23, %edx
	movl	$1, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$7, -16(%rbp)
	jmp	.L104
.L95:
	call	__errno_location@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$0, (%rax)
	movq	$14, -16(%rbp)
	jmp	.L104
.L100:
	movl	$10, %edi
	call	putchar@PLT
	movq	$6, -16(%rbp)
	jmp	.L104
.L87:
	call	__errno_location@PLT
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L104
.L93:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$6, -16(%rbp)
	jmp	.L104
.L91:
	cmpq	$0, -32(%rbp)
	je	.L106
	movq	$0, -16(%rbp)
	jmp	.L104
.L106:
	movq	$16, -16(%rbp)
	jmp	.L104
.L97:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$3, -16(%rbp)
	jmp	.L104
.L94:
	cmpq	$0, -40(%rbp)
	jne	.L108
	movq	$12, -16(%rbp)
	jmp	.L104
.L108:
	movq	$8, -16(%rbp)
	jmp	.L104
.L101:
	movq	-32(%rbp), %rax
	addq	$19, %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$14, -16(%rbp)
	jmp	.L104
.L111:
	nop
.L104:
	jmp	.L110
.L112:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	showlist, .-showlist
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
