	.file	"iains_gcc-14-branch_fd-glibc-byte-stream-connection-server_flatten.c"
	.text
	.globl	_TIG_IZ_o0C0_argv
	.bss
	.align 8
	.type	_TIG_IZ_o0C0_argv, @object
	.size	_TIG_IZ_o0C0_argv, 8
_TIG_IZ_o0C0_argv:
	.zero	8
	.globl	_TIG_IZ_o0C0_argc
	.align 4
	.type	_TIG_IZ_o0C0_argc, @object
	.size	_TIG_IZ_o0C0_argc, 4
_TIG_IZ_o0C0_argc:
	.zero	4
	.globl	_TIG_IZ_o0C0_envp
	.align 8
	.type	_TIG_IZ_o0C0_envp, @object
	.size	_TIG_IZ_o0C0_envp, 8
_TIG_IZ_o0C0_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Server: got message: `%s'\n"
.LC1:
	.string	"read"
	.text
	.globl	read_from_client
	.type	read_from_client, @function
read_from_client:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$576, %rsp
	movl	%edi, -564(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -544(%rbp)
.L18:
	cmpq	$9, -544(%rbp)
	ja	.L21
	movq	-544(%rbp), %rax
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
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	leaq	-528(%rbp), %rcx
	movl	-564(%rbp), %eax
	movl	$512, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -536(%rbp)
	movq	-536(%rbp), %rax
	movl	%eax, -548(%rbp)
	movq	$2, -544(%rbp)
	jmp	.L12
.L5:
	movq	$4, -544(%rbp)
	jmp	.L12
.L10:
	cmpl	$0, -548(%rbp)
	jne	.L13
	movq	$0, -544(%rbp)
	jmp	.L12
.L13:
	movq	$9, -544(%rbp)
	jmp	.L12
.L8:
	movl	$0, %eax
	jmp	.L19
.L3:
	movq	stderr(%rip), %rax
	leaq	-528(%rbp), %rdx
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -544(%rbp)
	jmp	.L12
.L11:
	movl	$-1, %eax
	jmp	.L19
.L6:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L9:
	cmpl	$0, -548(%rbp)
	jns	.L16
	movq	$7, -544(%rbp)
	jmp	.L12
.L16:
	movq	$1, -544(%rbp)
	jmp	.L12
.L21:
	nop
.L12:
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
.LFE0:
	.size	read_from_client, .-read_from_client
	.section	.rodata
.LC2:
	.string	"accept"
	.align 8
.LC3:
	.string	"Server: connect from host %s, port %hd.\n"
.LC4:
	.string	"select"
.LC5:
	.string	"listen"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$384, %rsp
	movl	%edi, -356(%rbp)
	movq	%rsi, -368(%rbp)
	movq	%rdx, -376(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_o0C0_envp(%rip)
	nop
.L23:
	movq	$0, _TIG_IZ_o0C0_argv(%rip)
	nop
.L24:
	movl	$0, _TIG_IZ_o0C0_argc(%rip)
	nop
	nop
.L25:
.L26:
#APP
# 101 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-o0C0--0
# 0 "" 2
#NO_APP
	movl	-356(%rbp), %eax
	movl	%eax, _TIG_IZ_o0C0_argc(%rip)
	movq	-368(%rbp), %rax
	movq	%rax, _TIG_IZ_o0C0_argv(%rip)
	movq	-376(%rbp), %rax
	movq	%rax, _TIG_IZ_o0C0_envp(%rip)
	nop
	movq	$22, -304(%rbp)
.L69:
	cmpq	$39, -304(%rbp)
	ja	.L71
	movq	-304(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L51-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L50-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L49-.L29
	.long	.L48-.L29
	.long	.L47-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L46-.L29
	.long	.L45-.L29
	.long	.L71-.L29
	.long	.L44-.L29
	.long	.L71-.L29
	.long	.L43-.L29
	.long	.L71-.L29
	.long	.L42-.L29
	.long	.L41-.L29
	.long	.L40-.L29
	.long	.L39-.L29
	.long	.L38-.L29
	.long	.L37-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L36-.L29
	.long	.L71-.L29
	.long	.L35-.L29
	.long	.L71-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L71-.L29
	.long	.L71-.L29
	.long	.L28-.L29
	.text
.L43:
	addl	$1, -336(%rbp)
	movq	$32, -304(%rbp)
	jmp	.L52
.L37:
	cmpl	$15, -328(%rbp)
	ja	.L53
	movq	$30, -304(%rbp)
	jmp	.L52
.L53:
	movq	$35, -304(%rbp)
	jmp	.L52
.L35:
	movq	-312(%rbp), %rax
	movl	-328(%rbp), %edx
	movq	$0, (%rax,%rdx,8)
	addl	$1, -328(%rbp)
	movq	$25, -304(%rbp)
	jmp	.L52
.L45:
	movl	-336(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	cltq
	movq	-144(%rbp,%rax,8), %rdx
	movl	-336(%rbp), %eax
	andl	$63, %eax
	movl	$1, %esi
	movl	%eax, %ecx
	salq	%cl, %rsi
	movq	%rsi, %rax
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L55
	movq	$21, -304(%rbp)
	jmp	.L52
.L55:
	movq	$18, -304(%rbp)
	jmp	.L52
.L49:
	leaq	-272(%rbp), %rax
	movq	%rax, -312(%rbp)
	movl	$0, -328(%rbp)
	movq	$25, -304(%rbp)
	jmp	.L52
.L39:
	cmpl	$0, -316(%rbp)
	jns	.L57
	movq	$36, -304(%rbp)
	jmp	.L52
.L57:
	movq	$18, -304(%rbp)
	jmp	.L52
.L44:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L38:
	movzwl	-286(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %edi
	call	ntohs@PLT
	movw	%ax, -346(%rbp)
	movl	-284(%rbp), %eax
	movl	%eax, %edi
	call	inet_ntoa@PLT
	movq	%rax, -296(%rbp)
	movzwl	-346(%rbp), %ecx
	movq	stderr(%rip), %rax
	movq	-296(%rbp), %rdx
	leaq	.LC3(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-320(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-272(%rbp,%rax,8), %rdx
	movl	-320(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -272(%rbp,%rax,8)
	movq	$18, -304(%rbp)
	jmp	.L52
.L41:
	movl	-336(%rbp), %eax
	cmpl	-340(%rbp), %eax
	jne	.L59
	movq	$5, -304(%rbp)
	jmp	.L52
.L59:
	movq	$9, -304(%rbp)
	jmp	.L52
.L30:
	movl	-336(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-336(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-272(%rbp,%rax,8), %rdx
	movl	-336(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	notq	%rax
	andq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -272(%rbp,%rax,8)
	movq	$18, -304(%rbp)
	jmp	.L52
.L48:
	movl	-336(%rbp), %eax
	movl	%eax, %edi
	call	read_from_client
	movl	%eax, -316(%rbp)
	movq	$23, -304(%rbp)
	jmp	.L52
.L46:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L34:
	cmpl	$1023, -336(%rbp)
	jg	.L61
	movq	$14, -304(%rbp)
	jmp	.L52
.L61:
	movq	$28, -304(%rbp)
	jmp	.L52
.L32:
	cmpl	$0, -332(%rbp)
	jns	.L63
	movq	$20, -304(%rbp)
	jmp	.L52
.L63:
	movq	$8, -304(%rbp)
	jmp	.L52
.L40:
	movq	$33, -304(%rbp)
	jmp	.L52
.L36:
	movq	-272(%rbp), %rax
	movq	-264(%rbp), %rdx
	movq	%rax, -144(%rbp)
	movq	%rdx, -136(%rbp)
	movq	-256(%rbp), %rax
	movq	-248(%rbp), %rdx
	movq	%rax, -128(%rbp)
	movq	%rdx, -120(%rbp)
	movq	-240(%rbp), %rax
	movq	-232(%rbp), %rdx
	movq	%rax, -112(%rbp)
	movq	%rdx, -104(%rbp)
	movq	-224(%rbp), %rax
	movq	-216(%rbp), %rdx
	movq	%rax, -96(%rbp)
	movq	%rdx, -88(%rbp)
	movq	-208(%rbp), %rax
	movq	-200(%rbp), %rdx
	movq	%rax, -80(%rbp)
	movq	%rdx, -72(%rbp)
	movq	-192(%rbp), %rax
	movq	-184(%rbp), %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	movq	-176(%rbp), %rax
	movq	-168(%rbp), %rdx
	movq	%rax, -48(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-160(%rbp), %rax
	movq	-152(%rbp), %rdx
	movq	%rax, -32(%rbp)
	movq	%rdx, -24(%rbp)
	leaq	-144(%rbp), %rax
	movl	$0, %r8d
	movl	$0, %ecx
	movl	$0, %edx
	movq	%rax, %rsi
	movl	$1024, %edi
	call	select@PLT
	movl	%eax, -324(%rbp)
	movq	$39, -304(%rbp)
	jmp	.L52
.L50:
	movl	$16, -344(%rbp)
	leaq	-344(%rbp), %rdx
	leaq	-288(%rbp), %rcx
	movl	-340(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -320(%rbp)
	movq	$10, -304(%rbp)
	jmp	.L52
.L33:
	movl	$5555, %edi
	call	make_socket
	movl	%eax, -340(%rbp)
	movl	-340(%rbp), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	%eax, -332(%rbp)
	movq	$34, -304(%rbp)
	jmp	.L52
.L47:
	cmpl	$0, -320(%rbp)
	jns	.L65
	movq	$16, -304(%rbp)
	jmp	.L52
.L65:
	movq	$24, -304(%rbp)
	jmp	.L52
.L51:
	movl	$0, -336(%rbp)
	movq	$32, -304(%rbp)
	jmp	.L52
.L28:
	cmpl	$0, -324(%rbp)
	jns	.L67
	movq	$13, -304(%rbp)
	jmp	.L52
.L67:
	movq	$0, -304(%rbp)
	jmp	.L52
.L31:
	movl	-340(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-272(%rbp,%rax,8), %rdx
	movl	-340(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -272(%rbp,%rax,8)
	movq	$28, -304(%rbp)
	jmp	.L52
.L42:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L71:
	nop
.L52:
	jmp	.L69
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC6:
	.string	"socket"
.LC7:
	.string	"bind"
	.text
	.globl	make_socket
	.type	make_socket, @function
make_socket:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, %eax
	movw	%ax, -52(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -40(%rbp)
.L88:
	cmpq	$9, -40(%rbp)
	ja	.L91
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L91-.L75
	.long	.L81-.L75
	.long	.L80-.L75
	.long	.L79-.L75
	.long	.L78-.L75
	.long	.L91-.L75
	.long	.L77-.L75
	.long	.L91-.L75
	.long	.L76-.L75
	.long	.L74-.L75
	.text
.L78:
	movw	$2, -32(%rbp)
	movzwl	-52(%rbp), %eax
	movl	%eax, %edi
	call	htons@PLT
	movw	%ax, -30(%rbp)
	movl	$0, %edi
	call	htonl@PLT
	movl	%eax, -28(%rbp)
	leaq	-32(%rbp), %rcx
	movl	-48(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	%eax, -44(%rbp)
	movq	$2, -40(%rbp)
	jmp	.L82
.L76:
	cmpl	$0, -48(%rbp)
	jns	.L83
	movq	$9, -40(%rbp)
	jmp	.L82
.L83:
	movq	$4, -40(%rbp)
	jmp	.L82
.L81:
	movl	-48(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L89
	jmp	.L90
.L79:
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -48(%rbp)
	movq	$8, -40(%rbp)
	jmp	.L82
.L74:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L77:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L80:
	cmpl	$0, -44(%rbp)
	jns	.L86
	movq	$6, -40(%rbp)
	jmp	.L82
.L86:
	movq	$1, -40(%rbp)
	jmp	.L82
.L91:
	nop
.L82:
	jmp	.L88
.L90:
	call	__stack_chk_fail@PLT
.L89:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	make_socket, .-make_socket
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
