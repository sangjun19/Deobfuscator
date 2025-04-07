	.file	"Avgilboa_OS---EX2_}_flatten.c"
	.text
	.globl	_TIG_IZ_6bHl_argc
	.bss
	.align 4
	.type	_TIG_IZ_6bHl_argc, @object
	.size	_TIG_IZ_6bHl_argc, 4
_TIG_IZ_6bHl_argc:
	.zero	4
	.globl	_TIG_IZ_6bHl_envp
	.align 8
	.type	_TIG_IZ_6bHl_envp, @object
	.size	_TIG_IZ_6bHl_envp, 8
_TIG_IZ_6bHl_envp:
	.zero	8
	.globl	_TIG_IZ_6bHl_argv
	.align 8
	.type	_TIG_IZ_6bHl_argv, @object
	.size	_TIG_IZ_6bHl_argv, 8
_TIG_IZ_6bHl_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"This process the unix command: %s"
	.text
	.globl	unix_command
	.type	unix_command, @function
unix_command:
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
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
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
.LFE3:
	.size	unix_command, .-unix_command
	.section	.rodata
.LC1:
	.string	"pipe"
.LC2:
	.string	"%s"
	.text
	.globl	func
	.type	func, @function
func:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -164(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -120(%rbp)
.L39:
	cmpq	$26, -120(%rbp)
	ja	.L41
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L12(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L12(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L12:
	.long	.L27-.L12
	.long	.L26-.L12
	.long	.L25-.L12
	.long	.L24-.L12
	.long	.L23-.L12
	.long	.L22-.L12
	.long	.L41-.L12
	.long	.L21-.L12
	.long	.L20-.L12
	.long	.L41-.L12
	.long	.L19-.L12
	.long	.L41-.L12
	.long	.L41-.L12
	.long	.L18-.L12
	.long	.L41-.L12
	.long	.L41-.L12
	.long	.L41-.L12
	.long	.L17-.L12
	.long	.L16-.L12
	.long	.L41-.L12
	.long	.L41-.L12
	.long	.L15-.L12
	.long	.L14-.L12
	.long	.L41-.L12
	.long	.L13-.L12
	.long	.L41-.L12
	.long	.L11-.L12
	.text
.L16:
	movl	-100(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	dup2@PLT
	leaq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	unix_command
	movl	-104(%rbp), %eax
	leaq	-96(%rbp), %rcx
	movl	$80, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	leaq	-96(%rbp), %rcx
	movl	-164(%rbp), %eax
	movl	$80, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$21, -120(%rbp)
	jmp	.L28
.L23:
	call	fork@PLT
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -148(%rbp)
	movq	$26, -120(%rbp)
	jmp	.L28
.L20:
	cmpq	$0, -128(%rbp)
	jle	.L29
	movq	$17, -120(%rbp)
	jmp	.L28
.L29:
	movq	$21, -120(%rbp)
	jmp	.L28
.L26:
	movq	$21, -120(%rbp)
	jmp	.L28
.L24:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$18, -120(%rbp)
	jmp	.L28
.L13:
	movl	-156(%rbp), %eax
	movl	%eax, -140(%rbp)
	addl	$1, -156(%rbp)
	call	getchar@PLT
	movl	%eax, -136(%rbp)
	movl	-136(%rbp), %eax
	movb	%al, -157(%rbp)
	movl	-140(%rbp), %eax
	cltq
	movzbl	-157(%rbp), %edx
	movb	%dl, -96(%rbp,%rax)
	movq	$2, -120(%rbp)
	jmp	.L28
.L15:
	leaq	-96(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, 24(%rax)
	movq	$0, 32(%rax)
	movq	$0, 40(%rax)
	movq	$0, 48(%rax)
	movq	$0, 56(%rax)
	movq	$0, 64(%rax)
	movq	$0, 72(%rax)
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	pipe@PLT
	movl	%eax, -152(%rbp)
	movq	$22, -120(%rbp)
	jmp	.L28
.L11:
	cmpl	$0, -148(%rbp)
	jne	.L31
	movq	$13, -120(%rbp)
	jmp	.L28
.L31:
	movq	$5, -120(%rbp)
	jmp	.L28
.L18:
	movl	-108(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	-108(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-112(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	leaq	-96(%rbp), %rcx
	movl	-164(%rbp), %eax
	movl	$80, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -128(%rbp)
	movq	$8, -120(%rbp)
	jmp	.L28
.L17:
	leaq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-96(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, 24(%rax)
	movq	$0, 32(%rax)
	movq	$0, 40(%rax)
	movq	$0, 48(%rax)
	movq	$0, 56(%rax)
	movq	$0, 64(%rax)
	movq	$0, 72(%rax)
	movq	$21, -120(%rbp)
	jmp	.L28
.L14:
	cmpl	$0, -152(%rbp)
	jns	.L33
	movq	$10, -120(%rbp)
	jmp	.L28
.L33:
	movq	$4, -120(%rbp)
	jmp	.L28
.L22:
	movl	-112(%rbp), %eax
	movl	$0, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	-108(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-112(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	leaq	-96(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, 8(%rax)
	movq	$0, 16(%rax)
	movq	$0, 24(%rax)
	movq	$0, 32(%rax)
	movq	$0, 40(%rax)
	movq	$0, 48(%rax)
	movq	$0, 56(%rax)
	movq	$0, 64(%rax)
	movq	$0, 72(%rax)
	movl	$0, -156(%rbp)
	movq	$24, -120(%rbp)
	jmp	.L28
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$2, %edi
	call	exit@PLT
.L27:
	leaq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pipe@PLT
	movl	%eax, -144(%rbp)
	movq	$7, -120(%rbp)
	jmp	.L28
.L21:
	cmpl	$0, -144(%rbp)
	jns	.L35
	movq	$3, -120(%rbp)
	jmp	.L28
.L35:
	movq	$18, -120(%rbp)
	jmp	.L28
.L25:
	cmpb	$10, -157(%rbp)
	je	.L37
	movq	$24, -120(%rbp)
	jmp	.L28
.L37:
	movq	$0, -120(%rbp)
	jmp	.L28
.L41:
	nop
.L28:
	jmp	.L39
	.cfi_endproc
.LFE4:
	.size	func, .-func
	.section	.rodata
.LC3:
	.string	"listening.."
.LC4:
	.string	"server accept failed..."
.LC5:
	.string	"Usage : ./ncL [port] \n"
.LC6:
	.string	"listen to [any] in port %s \n"
.LC7:
	.string	"Listen failed..."
.LC8:
	.string	"socket bind failed..."
.LC9:
	.string	"socket creation failed..."
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
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_6bHl_envp(%rip)
	nop
.L43:
	movq	$0, _TIG_IZ_6bHl_argv(%rip)
	nop
.L44:
	movl	$0, _TIG_IZ_6bHl_argc(%rip)
	nop
	nop
.L45:
.L46:
#APP
# 134 "Avgilboa_OS---EX2_}.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-6bHl--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_6bHl_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_6bHl_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_6bHl_envp(%rip)
	nop
	movq	$7, -56(%rbp)
.L79:
	cmpq	$26, -56(%rbp)
	ja	.L82
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L49(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L49(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L49:
	.long	.L82-.L49
	.long	.L82-.L49
	.long	.L66-.L49
	.long	.L65-.L49
	.long	.L82-.L49
	.long	.L82-.L49
	.long	.L64-.L49
	.long	.L63-.L49
	.long	.L62-.L49
	.long	.L61-.L49
	.long	.L82-.L49
	.long	.L82-.L49
	.long	.L60-.L49
	.long	.L59-.L49
	.long	.L58-.L49
	.long	.L82-.L49
	.long	.L82-.L49
	.long	.L57-.L49
	.long	.L56-.L49
	.long	.L55-.L49
	.long	.L54-.L49
	.long	.L53-.L49
	.long	.L52-.L49
	.long	.L82-.L49
	.long	.L51-.L49
	.long	.L50-.L49
	.long	.L48-.L49
	.text
.L56:
	cmpl	$0, -76(%rbp)
	jns	.L67
	movq	$12, -56(%rbp)
	jmp	.L69
.L67:
	movq	$9, -56(%rbp)
	jmp	.L69
.L50:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -56(%rbp)
	jmp	.L69
.L58:
	cmpl	$-1, -80(%rbp)
	jne	.L70
	movq	$20, -56(%rbp)
	jmp	.L69
.L70:
	movq	$21, -56(%rbp)
	jmp	.L69
.L60:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L62:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L80
	jmp	.L81
.L65:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L51:
	cmpl	$0, -72(%rbp)
	je	.L73
	movq	$6, -56(%rbp)
	jmp	.L69
.L73:
	movq	$17, -56(%rbp)
	jmp	.L69
.L53:
	leaq	-48(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, 8(%rax)
	movw	$2, -48(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, -44(%rbp)
	movzwl	-90(%rbp), %eax
	movw	%ax, -46(%rbp)
	leaq	-48(%rbp), %rcx
	movl	-80(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	%eax, -72(%rbp)
	movq	$24, -56(%rbp)
	jmp	.L69
.L48:
	cmpl	$0, -68(%rbp)
	je	.L75
	movq	$13, -56(%rbp)
	jmp	.L69
.L75:
	movq	$25, -56(%rbp)
	jmp	.L69
.L61:
	movq	-112(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$19, -56(%rbp)
	jmp	.L69
.L59:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L55:
	movl	-76(%rbp), %eax
	movl	%eax, %edi
	call	func
	movl	-80(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$8, -56(%rbp)
	jmp	.L69
.L57:
	movl	-80(%rbp), %eax
	movl	$5, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	%eax, -68(%rbp)
	movq	$26, -56(%rbp)
	jmp	.L69
.L64:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L52:
	movl	$16, -88(%rbp)
	leaq	-88(%rbp), %rdx
	leaq	-32(%rbp), %rcx
	movl	-80(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -76(%rbp)
	movq	$18, -56(%rbp)
	jmp	.L69
.L63:
	cmpl	$2, -100(%rbp)
	je	.L77
	movq	$3, -56(%rbp)
	jmp	.L69
.L77:
	movq	$2, -56(%rbp)
	jmp	.L69
.L66:
	movq	-112(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -64(%rbp)
	movl	-64(%rbp), %eax
	movw	%ax, -90(%rbp)
	movl	$0, %edi
	call	htonl@PLT
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, -84(%rbp)
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -80(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L69
.L54:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L82:
	nop
.L69:
	jmp	.L79
.L81:
	call	__stack_chk_fail@PLT
.L80:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
