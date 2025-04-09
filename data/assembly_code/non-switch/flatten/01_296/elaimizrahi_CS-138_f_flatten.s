	.file	"elaimizrahi_CS-138_f_flatten.c"
	.text
	.globl	_TIG_IZ_1P6Y_argv
	.bss
	.align 8
	.type	_TIG_IZ_1P6Y_argv, @object
	.size	_TIG_IZ_1P6Y_argv, 8
_TIG_IZ_1P6Y_argv:
	.zero	8
	.globl	_TIG_IZ_1P6Y_argc
	.align 4
	.type	_TIG_IZ_1P6Y_argc, @object
	.size	_TIG_IZ_1P6Y_argc, 4
_TIG_IZ_1P6Y_argc:
	.zero	4
	.globl	_TIG_IZ_1P6Y_envp
	.align 8
	.type	_TIG_IZ_1P6Y_envp, @object
	.size	_TIG_IZ_1P6Y_envp, 8
_TIG_IZ_1P6Y_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"hi"
	.text
	.globl	vlintegerDestroy
	.type	vlintegerDestroy, @function
vlintegerDestroy:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L11
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$2, -8(%rbp)
	je	.L5
	jmp	.L12
.L4:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$3, -8(%rbp)
	jmp	.L7
.L5:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L8
	movq	$0, -8(%rbp)
	jmp	.L7
.L8:
	movq	$3, -8(%rbp)
	jmp	.L7
.L12:
	nop
.L7:
	jmp	.L10
.L11:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	vlintegerDestroy, .-vlintegerDestroy
	.section	.rodata
.LC1:
	.string	"length=%d\n"
.LC2:
	.string	"%d"
	.text
	.globl	vlintegerPrint
	.type	vlintegerPrint, @function
vlintegerPrint:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L26:
	cmpq	$8, -8(%rbp)
	ja	.L27
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L16(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L16(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L16:
	.long	.L27-.L16
	.long	.L21-.L16
	.long	.L20-.L16
	.long	.L28-.L16
	.long	.L18-.L16
	.long	.L27-.L16
	.long	.L17-.L16
	.long	.L27-.L16
	.long	.L15-.L16
	.text
.L18:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L22
.L15:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L23
	movq	$6, -8(%rbp)
	jmp	.L22
.L23:
	movq	$2, -8(%rbp)
	jmp	.L22
.L21:
	movq	$4, -8(%rbp)
	jmp	.L22
.L17:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L22
.L20:
	movl	$10, %edi
	call	putchar@PLT
	movq	$3, -8(%rbp)
	jmp	.L22
.L27:
	nop
.L22:
	jmp	.L26
.L28:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	vlintegerPrint, .-vlintegerPrint
	.section	.rodata
.LC3:
	.string	"%c"
.LC4:
	.string	"multiplication"
.LC5:
	.string	"main"
.LC6:
	.string	"elaimizrahi_CS-138_f.c"
.LC7:
	.string	"int2->arr[0] !=0"
.LC8:
	.string	"ARRARY: %d"
.LC9:
	.string	"add->arr[0] !=0"
.LC10:
	.string	"addition"
.LC11:
	.string	"int1->arr[0] !=0"
.LC12:
	.string	"mult->arr[0] !=0"
	.align 8
.LC13:
	.string	"Enter the digits separated by a space: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1P6Y_envp(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_1P6Y_argv(%rip)
	nop
.L31:
	movl	$0, _TIG_IZ_1P6Y_argc(%rip)
	nop
	nop
.L32:
.L33:
#APP
# 112 "elaimizrahi_CS-138_f.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1P6Y--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_1P6Y_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_1P6Y_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_1P6Y_envp(%rip)
	nop
	movq	$15, -48(%rbp)
.L66:
	cmpq	$22, -48(%rbp)
	ja	.L69
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L53-.L36
	.long	.L52-.L36
	.long	.L51-.L36
	.long	.L50-.L36
	.long	.L69-.L36
	.long	.L49-.L36
	.long	.L48-.L36
	.long	.L47-.L36
	.long	.L69-.L36
	.long	.L46-.L36
	.long	.L45-.L36
	.long	.L44-.L36
	.long	.L43-.L36
	.long	.L42-.L36
	.long	.L41-.L36
	.long	.L40-.L36
	.long	.L39-.L36
	.long	.L38-.L36
	.long	.L69-.L36
	.long	.L69-.L36
	.long	.L37-.L36
	.long	.L69-.L36
	.long	.L35-.L36
	.text
.L41:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	$2, -48(%rbp)
	jmp	.L54
.L40:
	movq	$10, -48(%rbp)
	jmp	.L54
.L43:
	movq	-80(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -84(%rbp)
	jge	.L55
	movq	$11, -48(%rbp)
	jmp	.L54
.L55:
	movq	$0, -48(%rbp)
	jmp	.L54
.L52:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerMult
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$17, -48(%rbp)
	jmp	.L54
.L50:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rcx
	movl	$319, %edx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L39:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L57
	movq	$6, -48(%rbp)
	jmp	.L54
.L57:
	movq	$3, -48(%rbp)
	jmp	.L54
.L44:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -84(%rbp)
	movq	$12, -48(%rbp)
	jmp	.L54
.L46:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	vlintegerCreate
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -84(%rbp)
	movq	$12, -48(%rbp)
	jmp	.L54
.L42:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rcx
	movl	$327, %edx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L38:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L59
	movq	$14, -48(%rbp)
	jmp	.L54
.L59:
	movq	$5, -48(%rbp)
	jmp	.L54
.L48:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerAdd
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$20, -48(%rbp)
	jmp	.L54
.L35:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rcx
	movl	$307, %edx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L49:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rcx
	movl	$333, %edx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L45:
	call	vlintegerCreate
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$7, -48(%rbp)
	jmp	.L54
.L53:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$16, -48(%rbp)
	jmp	.L54
.L47:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L61
	movq	$9, -48(%rbp)
	jmp	.L54
.L61:
	movq	$22, -48(%rbp)
	jmp	.L54
.L51:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L67
	jmp	.L68
.L37:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L64
	movq	$1, -48(%rbp)
	jmp	.L54
.L64:
	movq	$13, -48(%rbp)
	jmp	.L54
.L69:
	nop
.L54:
	jmp	.L66
.L68:
	call	__stack_chk_fail@PLT
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
.LC14:
	.string	"temp11: %d\n"
.LC15:
	.string	"mult: %d\n"
.LC16:
	.string	"temp1: %d\n"
.LC17:
	.string	"cleared: %d"
	.text
	.globl	vlintegerMult
	.type	vlintegerMult, @function
vlintegerMult:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$288, %rsp
	movq	%rdi, -280(%rbp)
	movq	%rsi, -288(%rbp)
	movq	$67, -112(%rbp)
.L187:
	cmpq	$129, -112(%rbp)
	ja	.L189
	movq	-112(%rbp), %rax
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
	.long	.L138-.L73
	.long	.L189-.L73
	.long	.L137-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L136-.L73
	.long	.L135-.L73
	.long	.L134-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L133-.L73
	.long	.L132-.L73
	.long	.L131-.L73
	.long	.L189-.L73
	.long	.L130-.L73
	.long	.L129-.L73
	.long	.L128-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L127-.L73
	.long	.L126-.L73
	.long	.L125-.L73
	.long	.L124-.L73
	.long	.L123-.L73
	.long	.L122-.L73
	.long	.L121-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L120-.L73
	.long	.L189-.L73
	.long	.L119-.L73
	.long	.L118-.L73
	.long	.L189-.L73
	.long	.L117-.L73
	.long	.L189-.L73
	.long	.L116-.L73
	.long	.L115-.L73
	.long	.L114-.L73
	.long	.L113-.L73
	.long	.L189-.L73
	.long	.L112-.L73
	.long	.L111-.L73
	.long	.L189-.L73
	.long	.L110-.L73
	.long	.L109-.L73
	.long	.L108-.L73
	.long	.L107-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L106-.L73
	.long	.L189-.L73
	.long	.L105-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L104-.L73
	.long	.L103-.L73
	.long	.L102-.L73
	.long	.L101-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L100-.L73
	.long	.L99-.L73
	.long	.L189-.L73
	.long	.L98-.L73
	.long	.L189-.L73
	.long	.L97-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L96-.L73
	.long	.L95-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L94-.L73
	.long	.L93-.L73
	.long	.L92-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L91-.L73
	.long	.L90-.L73
	.long	.L89-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L88-.L73
	.long	.L87-.L73
	.long	.L189-.L73
	.long	.L86-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L85-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L84-.L73
	.long	.L83-.L73
	.long	.L82-.L73
	.long	.L81-.L73
	.long	.L80-.L73
	.long	.L189-.L73
	.long	.L79-.L73
	.long	.L78-.L73
	.long	.L77-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L74-.L73
	.long	.L189-.L73
	.long	.L189-.L73
	.long	.L72-.L73
	.text
.L129:
	movq	-120(%rbp), %rdx
	movq	-144(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerAdd
	movq	%rax, -144(%rbp)
	movl	$0, -184(%rbp)
	movq	$5, -112(%rbp)
	jmp	.L139
.L72:
	movl	-260(%rbp), %eax
	subl	-256(%rbp), %eax
	cmpl	%eax, -232(%rbp)
	jge	.L140
	movq	$64, -112(%rbp)
	jmp	.L139
.L140:
	movq	$13, -112(%rbp)
	jmp	.L139
.L107:
	addl	$1, -220(%rbp)
	subl	$1, -216(%rbp)
	movq	$14, -112(%rbp)
	jmp	.L139
.L86:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	movl	%eax, -208(%rbp)
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movl	4(%rax), %eax
	movl	%eax, -204(%rbp)
	movl	$0, -200(%rbp)
	movq	$24, -112(%rbp)
	jmp	.L139
.L125:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-200(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -200(%rbp)
	movq	$24, -112(%rbp)
	jmp	.L139
.L108:
	addl	$1, -164(%rbp)
	movq	$45, -112(%rbp)
	jmp	.L139
.L87:
	addl	$1, -228(%rbp)
	subl	$1, -224(%rbp)
	movq	$19, -112(%rbp)
	jmp	.L139
.L132:
	cmpl	$0, -216(%rbp)
	js	.L142
	movq	$121, -112(%rbp)
	jmp	.L139
.L142:
	movq	$102, -112(%rbp)
	jmp	.L139
.L131:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -196(%rbp)
	jge	.L144
	movq	$29, -112(%rbp)
	jmp	.L139
.L144:
	movq	$115, -112(%rbp)
	jmp	.L139
.L95:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -176(%rbp)
	jge	.L146
	movq	$39, -112(%rbp)
	jmp	.L139
.L146:
	movq	$58, -112(%rbp)
	jmp	.L139
.L106:
	movq	-280(%rbp), %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L148
	movq	$47, -112(%rbp)
	jmp	.L139
.L148:
	movq	$2, -112(%rbp)
	jmp	.L139
.L77:
	movq	-280(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-224(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-216(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	imull	%edx, %eax
	movq	-136(%rbp), %rdx
	movq	8(%rdx), %rdx
	movslq	%eax, %rcx
	imulq	$1717986919, %rcx, %rcx
	shrq	$32, %rcx
	sarl	$2, %ecx
	sarl	$31, %eax
	movl	%eax, %esi
	movl	%ecx, %eax
	subl	%esi, %eax
	movl	%eax, (%rdx)
	movq	-280(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-224(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-216(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	leaq	4(%rax), %rsi
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %edi
	sarl	$31, %edi
	subl	%edi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, (%rsi)
	movl	$0, -212(%rbp)
	movq	$35, -112(%rbp)
	jmp	.L139
.L88:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -192(%rbp)
	jge	.L150
	movq	$116, -112(%rbp)
	jmp	.L139
.L150:
	movq	$88, -112(%rbp)
	jmp	.L139
.L91:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-188(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -188(%rbp)
	movq	$120, -112(%rbp)
	jmp	.L139
.L111:
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -164(%rbp)
	jge	.L152
	movq	$125, -112(%rbp)
	jmp	.L139
.L152:
	movq	$107, -112(%rbp)
	jmp	.L139
.L76:
	movl	-260(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -236(%rbp)
	movq	$114, -112(%rbp)
	jmp	.L139
.L96:
	movq	-280(%rbp), %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movl	%edx, (%rax)
	movq	$17, -112(%rbp)
	jmp	.L139
.L127:
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -260(%rbp)
	movq	-280(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -256(%rbp)
	movl	-260(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-280(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -32(%rbp)
	movq	-280(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -240(%rbp)
	movq	$66, -112(%rbp)
	jmp	.L139
.L100:
	movl	$0, -244(%rbp)
	movq	$113, -112(%rbp)
	jmp	.L139
.L126:
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -200(%rbp)
	jge	.L154
	movq	$25, -112(%rbp)
	jmp	.L139
.L154:
	movq	$98, -112(%rbp)
	jmp	.L139
.L124:
	movl	-260(%rbp), %eax
	subl	-256(%rbp), %eax
	cmpl	%eax, -252(%rbp)
	jge	.L156
	movq	$119, -112(%rbp)
	jmp	.L139
.L156:
	movq	$70, -112(%rbp)
	jmp	.L139
.L89:
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	call	vlintegerCreate
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -120(%rbp)
	movl	-220(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-120(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -80(%rbp)
	movq	-120(%rbp), %rax
	movq	-80(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-120(%rbp), %rax
	movl	-220(%rbp), %edx
	movl	%edx, (%rax)
	movq	-120(%rbp), %rax
	movq	8(%rax), %rax
	movl	-208(%rbp), %edx
	movl	%edx, (%rax)
	movq	-120(%rbp), %rax
	movq	8(%rax), %rax
	leaq	4(%rax), %rdx
	movl	-204(%rbp), %eax
	movl	%eax, (%rdx)
	movl	$2, -196(%rbp)
	movq	$15, -112(%rbp)
	jmp	.L139
.L133:
	movq	-136(%rbp), %rax
	movl	$2, (%rax)
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-136(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-128(%rbp), %rax
	movl	$2, (%rax)
	movq	-128(%rbp), %rax
	movq	8(%rax), %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-128(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-144(%rbp), %rax
	movq	8(%rax), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-144(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-144(%rbp), %rax
	movl	$1, (%rax)
	movl	$0, -148(%rbp)
	movl	$0, -228(%rbp)
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -224(%rbp)
	movq	$19, -112(%rbp)
	jmp	.L139
.L85:
	movl	-168(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -96(%rbp)
	movq	-144(%rbp), %rax
	movq	-96(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-144(%rbp), %rax
	movl	-168(%rbp), %edx
	movl	%edx, (%rax)
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-136(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-128(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$73, -112(%rbp)
	jmp	.L139
.L75:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-164(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L158
	movq	$126, -112(%rbp)
	jmp	.L139
.L158:
	movq	$49, -112(%rbp)
	jmp	.L139
.L128:
	cmpl	$0, -224(%rbp)
	js	.L160
	movq	$28, -112(%rbp)
	jmp	.L139
.L160:
	movq	$65, -112(%rbp)
	jmp	.L139
.L120:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-184(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -184(%rbp)
	movq	$5, -112(%rbp)
	jmp	.L139
.L130:
	movq	-288(%rbp), %rax
	movl	(%rax), %edx
	movq	-280(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L162
	movq	$27, -112(%rbp)
	jmp	.L139
.L162:
	movq	$56, -112(%rbp)
	jmp	.L139
.L115:
	movl	$0, -232(%rbp)
	movq	$129, -112(%rbp)
	jmp	.L139
.L101:
	movq	$44, -112(%rbp)
	jmp	.L139
.L80:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-172(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -172(%rbp)
	movq	$71, -112(%rbp)
	jmp	.L139
.L135:
	movl	$0, -180(%rbp)
	movq	$7, -112(%rbp)
	jmp	.L139
.L81:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-192(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -192(%rbp)
	movq	$101, -112(%rbp)
	jmp	.L139
.L123:
	movq	-288(%rbp), %rax
	movl	(%rax), %edx
	movq	-280(%rbp), %rax
	movl	%edx, (%rax)
	movq	$56, -112(%rbp)
	jmp	.L139
.L93:
	movq	-288(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-248(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-288(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-248(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	subl	$1, -248(%rbp)
	movq	$97, -112(%rbp)
	jmp	.L139
.L84:
	movl	-260(%rbp), %eax
	subl	-256(%rbp), %eax
	cmpl	%eax, -244(%rbp)
	jge	.L164
	movq	$41, -112(%rbp)
	jmp	.L139
.L164:
	movq	$2, -112(%rbp)
	jmp	.L139
.L105:
	movq	-120(%rbp), %rax
	movl	$2, (%rax)
	movq	-120(%rbp), %rax
	movq	8(%rax), %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -72(%rbp)
	movq	-120(%rbp), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -172(%rbp)
	movq	$71, -112(%rbp)
	jmp	.L139
.L119:
	addl	$1, -240(%rbp)
	movq	$66, -112(%rbp)
	jmp	.L139
.L97:
	movq	-280(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-236(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-280(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-236(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	subl	$1, -236(%rbp)
	movq	$114, -112(%rbp)
	jmp	.L139
.L109:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-212(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -212(%rbp)
	movq	$35, -112(%rbp)
	jmp	.L139
.L99:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -172(%rbp)
	jge	.L166
	movq	$117, -112(%rbp)
	jmp	.L139
.L166:
	movq	$50, -112(%rbp)
	jmp	.L139
.L122:
	movl	$0, -220(%rbp)
	movq	-280(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -216(%rbp)
	movq	$14, -112(%rbp)
	jmp	.L139
.L103:
	movl	$0, -168(%rbp)
	movl	$0, -164(%rbp)
	movq	$45, -112(%rbp)
	jmp	.L139
.L110:
	movq	-280(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -260(%rbp)
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -256(%rbp)
	movl	-260(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-288(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -64(%rbp)
	movq	-288(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -252(%rbp)
	movq	$26, -112(%rbp)
	jmp	.L139
.L98:
	movq	-144(%rbp), %rax
	jmp	.L188
.L112:
	call	vlintegerCreate
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -144(%rbp)
	call	vlintegerCreate
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -136(%rbp)
	call	vlintegerCreate
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -128(%rbp)
	movl	$0, -160(%rbp)
	movl	$1, -156(%rbp)
	movl	$2, -152(%rbp)
	movq	$42, -112(%rbp)
	jmp	.L139
.L136:
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -184(%rbp)
	jge	.L169
	movq	$32, -112(%rbp)
	jmp	.L139
.L169:
	movq	$6, -112(%rbp)
	jmp	.L139
.L78:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -188(%rbp)
	jge	.L171
	movq	$96, -112(%rbp)
	jmp	.L139
.L171:
	movq	$18, -112(%rbp)
	jmp	.L139
.L90:
	cmpl	$0, -248(%rbp)
	js	.L173
	movq	$87, -112(%rbp)
	jmp	.L139
.L173:
	movq	$37, -112(%rbp)
	jmp	.L139
.L83:
	cmpl	$0, -236(%rbp)
	js	.L175
	movq	$75, -112(%rbp)
	jmp	.L139
.L175:
	movq	$34, -112(%rbp)
	jmp	.L139
.L117:
	addl	$1, -252(%rbp)
	movq	$26, -112(%rbp)
	jmp	.L139
.L104:
	movq	-280(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-232(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -232(%rbp)
	movq	$129, -112(%rbp)
	jmp	.L139
.L79:
	movl	-260(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -248(%rbp)
	movq	$97, -112(%rbp)
	jmp	.L139
.L114:
	movq	-288(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-244(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -244(%rbp)
	movq	$113, -112(%rbp)
	jmp	.L139
.L82:
	movl	-220(%rbp), %eax
	leal	2(%rax), %edx
	movl	-228(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-120(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -104(%rbp)
	movq	-120(%rbp), %rax
	movq	-104(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-220(%rbp), %eax
	leal	2(%rax), %edx
	movl	-228(%rbp), %eax
	addl	%eax, %edx
	movq	-120(%rbp), %rax
	movl	%edx, (%rax)
	movl	$2, -192(%rbp)
	movq	$101, -112(%rbp)
	jmp	.L139
.L113:
	movq	-280(%rbp), %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L177
	movq	$81, -112(%rbp)
	jmp	.L139
.L177:
	movq	$17, -112(%rbp)
	jmp	.L139
.L138:
	movl	$0, -176(%rbp)
	movq	$82, -112(%rbp)
	jmp	.L139
.L116:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-176(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -176(%rbp)
	movq	$82, -112(%rbp)
	jmp	.L139
.L102:
	movl	-260(%rbp), %eax
	subl	-256(%rbp), %eax
	cmpl	%eax, -240(%rbp)
	jge	.L179
	movq	$124, -112(%rbp)
	jmp	.L139
.L179:
	movq	$40, -112(%rbp)
	jmp	.L139
.L134:
	movq	-120(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -180(%rbp)
	jge	.L181
	movq	$86, -112(%rbp)
	jmp	.L139
.L181:
	movq	$0, -112(%rbp)
	jmp	.L139
.L92:
	movl	$0, -188(%rbp)
	movq	$120, -112(%rbp)
	jmp	.L139
.L118:
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -212(%rbp)
	jge	.L183
	movq	$48, -112(%rbp)
	jmp	.L139
.L183:
	movq	$104, -112(%rbp)
	jmp	.L139
.L121:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-196(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -196(%rbp)
	movq	$15, -112(%rbp)
	jmp	.L139
.L74:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-164(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-168(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	addl	$1, -168(%rbp)
	movq	$49, -112(%rbp)
	jmp	.L139
.L94:
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-180(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	-120(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-180(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -180(%rbp)
	movq	$7, -112(%rbp)
	jmp	.L139
.L137:
	movq	-280(%rbp), %rax
	movl	(%rax), %edx
	movq	-288(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L185
	movq	$23, -112(%rbp)
	jmp	.L139
.L185:
	movq	$13, -112(%rbp)
	jmp	.L139
.L189:
	nop
.L139:
	jmp	.L187
.L188:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	vlintegerMult, .-vlintegerMult
	.globl	vlintegerRead
	.type	vlintegerRead, @function
vlintegerRead:
.LFB7:
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
	movq	$1, -32(%rbp)
.L203:
	cmpq	$9, -32(%rbp)
	ja	.L206
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L193(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L193(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L193:
	.long	.L198-.L193
	.long	.L197-.L193
	.long	.L196-.L193
	.long	.L206-.L193
	.long	.L206-.L193
	.long	.L206-.L193
	.long	.L206-.L193
	.long	.L195-.L193
	.long	.L194-.L193
	.long	.L207-.L193
	.text
.L194:
	cmpl	$0, -36(%rbp)
	je	.L199
	movq	$7, -32(%rbp)
	jmp	.L201
.L199:
	movq	$2, -32(%rbp)
	jmp	.L201
.L197:
	movl	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L201
.L198:
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -36(%rbp)
	movq	$8, -32(%rbp)
	jmp	.L201
.L195:
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-56(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -40(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L201
.L196:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-56(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movl	-40(%rbp), %edx
	movl	%edx, (%rax)
	movq	$9, -32(%rbp)
	jmp	.L201
.L206:
	nop
.L201:
	jmp	.L203
.L207:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L205
	call	__stack_chk_fail@PLT
.L205:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	vlintegerRead, .-vlintegerRead
	.globl	power
	.type	power, @function
power:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$6, -8(%rbp)
.L220:
	cmpq	$7, -8(%rbp)
	ja	.L222
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L211(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L211(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L211:
	.long	.L215-.L211
	.long	.L222-.L211
	.long	.L214-.L211
	.long	.L222-.L211
	.long	.L222-.L211
	.long	.L213-.L211
	.long	.L212-.L211
	.long	.L210-.L211
	.text
.L212:
	movq	$5, -8(%rbp)
	jmp	.L216
.L213:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L216
.L215:
	movl	-16(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L216
.L210:
	movl	-16(%rbp), %eax
	jmp	.L221
.L214:
	movl	-12(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L218
	movq	$0, -8(%rbp)
	jmp	.L216
.L218:
	movq	$7, -8(%rbp)
	jmp	.L216
.L222:
	nop
.L216:
	jmp	.L220
.L221:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	power, .-power
	.section	.rodata
.LC18:
	.string	"%d %d \n"
.LC19:
	.string	"t1:%d\n"
.LC20:
	.string	"t2:%d\n"
	.text
	.globl	vlintegerAdd
	.type	vlintegerAdd, @function
vlintegerAdd:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	$31, -64(%rbp)
.L313:
	cmpq	$81, -64(%rbp)
	ja	.L315
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L226(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L226(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L226:
	.long	.L274-.L226
	.long	.L273-.L226
	.long	.L272-.L226
	.long	.L315-.L226
	.long	.L271-.L226
	.long	.L270-.L226
	.long	.L315-.L226
	.long	.L269-.L226
	.long	.L268-.L226
	.long	.L267-.L226
	.long	.L266-.L226
	.long	.L265-.L226
	.long	.L264-.L226
	.long	.L263-.L226
	.long	.L315-.L226
	.long	.L262-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L261-.L226
	.long	.L260-.L226
	.long	.L259-.L226
	.long	.L258-.L226
	.long	.L315-.L226
	.long	.L257-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L256-.L226
	.long	.L255-.L226
	.long	.L254-.L226
	.long	.L253-.L226
	.long	.L252-.L226
	.long	.L251-.L226
	.long	.L250-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L249-.L226
	.long	.L248-.L226
	.long	.L247-.L226
	.long	.L246-.L226
	.long	.L245-.L226
	.long	.L315-.L226
	.long	.L244-.L226
	.long	.L315-.L226
	.long	.L243-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L242-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L241-.L226
	.long	.L240-.L226
	.long	.L239-.L226
	.long	.L315-.L226
	.long	.L238-.L226
	.long	.L315-.L226
	.long	.L315-.L226
	.long	.L237-.L226
	.long	.L315-.L226
	.long	.L236-.L226
	.long	.L315-.L226
	.long	.L235-.L226
	.long	.L315-.L226
	.long	.L234-.L226
	.long	.L315-.L226
	.long	.L233-.L226
	.long	.L315-.L226
	.long	.L232-.L226
	.long	.L231-.L226
	.long	.L230-.L226
	.long	.L315-.L226
	.long	.L229-.L226
	.long	.L315-.L226
	.long	.L228-.L226
	.long	.L315-.L226
	.long	.L227-.L226
	.long	.L225-.L226
	.text
.L242:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%eax, %edx
	movl	-128(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -124(%rbp)
	movl	$0, -128(%rbp)
	movq	$29, -64(%rbp)
	jmp	.L275
.L227:
	movl	-92(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L276
	movq	$68, -64(%rbp)
	jmp	.L275
.L276:
	movq	$56, -64(%rbp)
	jmp	.L275
.L257:
	subl	$1, -100(%rbp)
	movq	$57, -64(%rbp)
	jmp	.L275
.L271:
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-152(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-160(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$7, -64(%rbp)
	jmp	.L275
.L254:
	movq	-72(%rbp), %rax
	movl	-132(%rbp), %edx
	movl	%edx, (%rax)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -32(%rbp)
	movq	-72(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -84(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L275
.L237:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$25, -64(%rbp)
	jmp	.L275
.L262:
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -100(%rbp)
	movq	$57, -64(%rbp)
	jmp	.L275
.L240:
	movl	$0, -88(%rbp)
	movq	$73, -64(%rbp)
	jmp	.L275
.L253:
	movq	$8, -64(%rbp)
	jmp	.L275
.L264:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -96(%rbp)
	jge	.L278
	movq	$2, -64(%rbp)
	jmp	.L275
.L278:
	movq	$0, -64(%rbp)
	jmp	.L275
.L268:
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movl	$0, -76(%rbp)
	movl	$0, -128(%rbp)
	call	vlintegerCreate
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$45, -64(%rbp)
	jmp	.L275
.L243:
	movq	-152(%rbp), %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L280
	movq	$72, -64(%rbp)
	jmp	.L275
.L280:
	movq	$70, -64(%rbp)
	jmp	.L275
.L228:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	-128(%rbp), %edx
	movl	%edx, (%rax)
	movq	$4, -64(%rbp)
	jmp	.L275
.L273:
	cmpl	$0, -100(%rbp)
	jne	.L282
	movq	$13, -64(%rbp)
	jmp	.L275
.L282:
	movq	$21, -64(%rbp)
	jmp	.L275
.L225:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	subl	$1, -80(%rbp)
	movq	$64, -64(%rbp)
	jmp	.L275
.L258:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$5, -64(%rbp)
	jmp	.L275
.L233:
	movq	-152(%rbp), %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L284
	movq	$40, -64(%rbp)
	jmp	.L275
.L284:
	movq	$0, -64(%rbp)
	jmp	.L275
.L260:
	cmpl	$0, -100(%rbp)
	je	.L286
	movq	$62, -64(%rbp)
	jmp	.L275
.L286:
	movq	$25, -64(%rbp)
	jmp	.L275
.L229:
	movl	$0, -96(%rbp)
	movq	$12, -64(%rbp)
	jmp	.L275
.L239:
	cmpl	$0, -100(%rbp)
	js	.L288
	movq	$1, -64(%rbp)
	jmp	.L275
.L288:
	movq	$43, -64(%rbp)
	jmp	.L275
.L234:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -92(%rbp)
	movq	$80, -64(%rbp)
	jmp	.L275
.L265:
	cmpl	$0, -84(%rbp)
	js	.L290
	movq	$50, -64(%rbp)
	jmp	.L275
.L290:
	movq	$28, -64(%rbp)
	jmp	.L275
.L267:
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -112(%rbp)
	movq	$22, -64(%rbp)
	jmp	.L275
.L263:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$21, -64(%rbp)
	jmp	.L275
.L252:
	movl	-124(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -128(%rbp)
	movl	-124(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%edx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %ecx
	movl	%ecx, %eax
	sall	$2, %eax
	addl	%ecx, %eax
	addl	%eax, %eax
	subl	%eax, %edx
	movl	%edx, -124(%rbp)
	movq	$41, -64(%rbp)
	jmp	.L275
.L246:
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -120(%rbp)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -48(%rbp)
	movq	-152(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -104(%rbp)
	movq	$74, -64(%rbp)
	jmp	.L275
.L241:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -116(%rbp)
	jge	.L292
	movq	$9, -64(%rbp)
	jmp	.L275
.L292:
	movq	$39, -64(%rbp)
	jmp	.L275
.L238:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$10, -64(%rbp)
	jmp	.L275
.L248:
	cmpl	$0, -112(%rbp)
	jne	.L294
	movq	$23, -64(%rbp)
	jmp	.L275
.L294:
	movq	$5, -64(%rbp)
	jmp	.L275
.L250:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jge	.L296
	movq	$33, -64(%rbp)
	jmp	.L275
.L296:
	movq	$70, -64(%rbp)
	jmp	.L275
.L230:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -104(%rbp)
	jge	.L298
	movq	$15, -64(%rbp)
	jmp	.L275
.L298:
	movq	$76, -64(%rbp)
	jmp	.L275
.L259:
	cmpl	$0, -112(%rbp)
	js	.L300
	movq	$38, -64(%rbp)
	jmp	.L275
.L300:
	movq	$20, -64(%rbp)
	jmp	.L275
.L256:
	cmpl	$0, -128(%rbp)
	jle	.L302
	movq	$37, -64(%rbp)
	jmp	.L275
.L302:
	movq	$4, -64(%rbp)
	jmp	.L275
.L231:
	movl	-88(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L304
	movq	$66, -64(%rbp)
	jmp	.L275
.L304:
	movq	$30, -64(%rbp)
	jmp	.L275
.L270:
	cmpl	$0, -112(%rbp)
	je	.L306
	movq	$59, -64(%rbp)
	jmp	.L275
.L306:
	movq	$10, -64(%rbp)
	jmp	.L275
.L232:
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -120(%rbp)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -56(%rbp)
	movq	-160(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -116(%rbp)
	movq	$55, -64(%rbp)
	jmp	.L275
.L251:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-108(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -108(%rbp)
	movq	$34, -64(%rbp)
	jmp	.L275
.L249:
	movl	-132(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -40(%rbp)
	movq	-72(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	leal	1(%rax), %edx
	movq	-72(%rbp), %rax
	movl	%edx, (%rax)
	movl	-132(%rbp), %eax
	movl	%eax, -80(%rbp)
	movq	$64, -64(%rbp)
	jmp	.L275
.L236:
	cmpl	$0, -80(%rbp)
	jle	.L308
	movq	$81, -64(%rbp)
	jmp	.L275
.L308:
	movq	$78, -64(%rbp)
	jmp	.L275
.L245:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-124(%rbp), %eax
	movl	%eax, (%rdx)
	subl	$1, -84(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L275
.L266:
	subl	$1, -112(%rbp)
	movq	$22, -64(%rbp)
	jmp	.L275
.L274:
	movl	$0, -92(%rbp)
	movq	$80, -64(%rbp)
	jmp	.L275
.L247:
	movl	$0, -108(%rbp)
	movq	$34, -64(%rbp)
	jmp	.L275
.L235:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-88(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -88(%rbp)
	movq	$73, -64(%rbp)
	jmp	.L275
.L269:
	movq	-72(%rbp), %rax
	jmp	.L314
.L255:
	cmpl	$9, -124(%rbp)
	jle	.L311
	movq	$32, -64(%rbp)
	jmp	.L275
.L311:
	movq	$41, -64(%rbp)
	jmp	.L275
.L244:
	addl	$1, -104(%rbp)
	movq	$74, -64(%rbp)
	jmp	.L275
.L272:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-96(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -96(%rbp)
	movq	$12, -64(%rbp)
	jmp	.L275
.L261:
	addl	$1, -116(%rbp)
	movq	$55, -64(%rbp)
	jmp	.L275
.L315:
	nop
.L275:
	jmp	.L313
.L314:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	vlintegerAdd, .-vlintegerAdd
	.globl	vlintegerCreate
	.type	vlintegerCreate, @function
vlintegerCreate:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -16(%rbp)
.L322:
	cmpq	$2, -16(%rbp)
	je	.L317
	cmpq	$2, -16(%rbp)
	ja	.L324
	cmpq	$0, -16(%rbp)
	je	.L319
	cmpq	$1, -16(%rbp)
	jne	.L324
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$0, (%rax)
	movq	$0, -16(%rbp)
	jmp	.L320
.L319:
	movq	-24(%rbp), %rax
	jmp	.L323
.L317:
	movq	$1, -16(%rbp)
	jmp	.L320
.L324:
	nop
.L320:
	jmp	.L322
.L323:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	vlintegerCreate, .-vlintegerCreate
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
