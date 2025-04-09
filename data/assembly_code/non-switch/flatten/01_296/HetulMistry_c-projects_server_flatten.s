	.file	"HetulMistry_c-projects_server_flatten.c"
	.text
	.globl	_TIG_IZ_HjDX_argc
	.bss
	.align 4
	.type	_TIG_IZ_HjDX_argc, @object
	.size	_TIG_IZ_HjDX_argc, 4
_TIG_IZ_HjDX_argc:
	.zero	4
	.globl	_TIG_IZ_HjDX_envp
	.align 8
	.type	_TIG_IZ_HjDX_envp, @object
	.size	_TIG_IZ_HjDX_envp, 8
_TIG_IZ_HjDX_envp:
	.zero	8
	.globl	_TIG_IZ_HjDX_argv
	.align 8
	.type	_TIG_IZ_HjDX_argv, @object
	.size	_TIG_IZ_HjDX_argv, 8
_TIG_IZ_HjDX_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Error listening for connections"
.LC1:
	.string	"Error accepting connection"
.LC2:
	.string	"Error binding socket"
.LC3:
	.string	"Client connected"
.LC4:
	.string	"Error creating socket"
	.align 8
.LC5:
	.string	"Server is listening on port %d...\n"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_HjDX_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_HjDX_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_HjDX_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 151 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HjDX--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_HjDX_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_HjDX_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_HjDX_envp(%rip)
	nop
	movq	$19, -56(%rbp)
.L37:
	cmpq	$23, -56(%rbp)
	ja	.L39
	movq	-56(%rbp), %rax
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
	.long	.L25-.L8
	.long	.L39-.L8
	.long	.L24-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L23-.L8
	.long	.L39-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-72(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$10, -56(%rbp)
	jmp	.L26
.L15:
	cmpl	$-1, -64(%rbp)
	jne	.L27
	movq	$15, -56(%rbp)
	jmp	.L26
.L27:
	movq	$5, -56(%rbp)
	jmp	.L26
.L14:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$5, -56(%rbp)
	jmp	.L26
.L17:
	cmpl	$-1, -68(%rbp)
	jne	.L29
	movq	$11, -56(%rbp)
	jmp	.L26
.L29:
	movq	$7, -56(%rbp)
	jmp	.L26
.L21:
	cmpl	$-1, -76(%rbp)
	jne	.L31
	movq	$13, -56(%rbp)
	jmp	.L26
.L31:
	movq	$23, -56(%rbp)
	jmp	.L26
.L7:
	leaq	-48(%rbp), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$2, -48(%rbp)
	movl	$0, -44(%rbp)
	movl	$8888, %edi
	call	htons@PLT
	movw	%ax, -46(%rbp)
	leaq	-48(%rbp), %rcx
	movl	-76(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	%eax, -68(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L26
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$9, -56(%rbp)
	jmp	.L26
.L18:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$7, -56(%rbp)
	jmp	.L26
.L20:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	fork@PLT
	movl	%eax, -60(%rbp)
	movq	$0, -56(%rbp)
	jmp	.L26
.L16:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	error
	movq	$23, -56(%rbp)
	jmp	.L26
.L12:
	movq	$22, -56(%rbp)
	jmp	.L26
.L9:
	movl	$16, -80(%rbp)
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -76(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L26
.L23:
	movl	$8888, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -56(%rbp)
	jmp	.L26
.L19:
	leaq	-80(%rbp), %rdx
	leaq	-32(%rbp), %rcx
	movl	-76(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -72(%rbp)
	movq	$20, -56(%rbp)
	jmp	.L26
.L25:
	cmpl	$0, -60(%rbp)
	jne	.L33
	movq	$2, -56(%rbp)
	jmp	.L26
.L33:
	movq	$18, -56(%rbp)
	jmp	.L26
.L22:
	movl	-76(%rbp), %eax
	movl	$5, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	%eax, -64(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L26
.L24:
	movl	-76(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-72(%rbp), %eax
	movl	%eax, %edi
	call	monitor_network
	movl	-72(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$0, %edi
	call	exit@PLT
.L11:
	cmpl	$-1, -72(%rbp)
	jne	.L35
	movq	$21, -56(%rbp)
	jmp	.L26
.L35:
	movq	$9, -56(%rbp)
	jmp	.L26
.L39:
	nop
.L26:
	jmp	.L37
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC6:
	.string	"Bandwidth: %d KB/s\nConnected Devices: %d\nNetwork Statistics: ...\n"
	.text
	.globl	monitor_network
	.type	monitor_network, @function
monitor_network:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1088, %rsp
	movl	%edi, -1076(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -1056(%rbp)
.L44:
	cmpq	$0, -1056(%rbp)
	je	.L41
	cmpq	$1, -1056(%rbp)
	jne	.L46
	call	rand@PLT
	movl	%eax, -1064(%rbp)
	call	rand@PLT
	movl	%eax, -1060(%rbp)
	movl	-1064(%rbp), %edx
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
	movl	%edx, %ecx
	subl	%eax, %ecx
	movl	-1060(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$5, %eax
	movl	%edx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	imull	$100, %eax, %esi
	movl	%edx, %eax
	subl	%esi, %eax
	leaq	-1040(%rbp), %rdi
	movl	%ecx, %r8d
	movl	%eax, %ecx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdx
	movl	$1024, %esi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rdx
	leaq	-1040(%rbp), %rsi
	movl	-1076(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movl	$1, %edi
	call	sleep@PLT
	movq	$1, -1056(%rbp)
	jmp	.L43
.L41:
	movq	$1, -1056(%rbp)
	jmp	.L43
.L46:
	nop
.L43:
	jmp	.L44
	.cfi_endproc
.LFE7:
	.size	monitor_network, .-monitor_network
	.globl	error
	.type	error, @function
error:
.LFB8:
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
.L51:
	cmpq	$0, -8(%rbp)
	je	.L48
	cmpq	$1, -8(%rbp)
	jne	.L52
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L48:
	movq	$1, -8(%rbp)
	jmp	.L50
.L52:
	nop
.L50:
	jmp	.L51
	.cfi_endproc
.LFE8:
	.size	error, .-error
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
