	.file	"FeckingPotato_EltexIPC_main_flatten.c"
	.text
	.globl	_TIG_IZ_NMNL_argc
	.bss
	.align 4
	.type	_TIG_IZ_NMNL_argc, @object
	.size	_TIG_IZ_NMNL_argc, 4
_TIG_IZ_NMNL_argc:
	.zero	4
	.globl	_TIG_IZ_NMNL_argv
	.align 8
	.type	_TIG_IZ_NMNL_argv, @object
	.size	_TIG_IZ_NMNL_argv, 8
_TIG_IZ_NMNL_argv:
	.zero	8
	.globl	_TIG_IZ_NMNL_envp
	.align 8
	.type	_TIG_IZ_NMNL_envp, @object
	.size	_TIG_IZ_NMNL_envp, 8
_TIG_IZ_NMNL_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"fork"
.LC1:
	.string	"%d "
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$0, _TIG_IZ_NMNL_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_NMNL_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_NMNL_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 152 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NMNL--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_NMNL_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_NMNL_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_NMNL_envp(%rip)
	nop
	movq	$18, -40(%rbp)
.L27:
	cmpq	$19, -40(%rbp)
	ja	.L28
	movq	-40(%rbp), %rax
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
	.long	.L28-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	call	fork@PLT
	movl	%eax, -52(%rbp)
	movq	$17, -40(%rbp)
	jmp	.L19
.L11:
	movl	$1, -48(%rbp)
	movq	$10, -40(%rbp)
	jmp	.L19
.L14:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L18:
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, %edi
	call	exit@PLT
.L12:
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, %edi
	call	exit@PLT
.L7:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	$10, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtol@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	imulq	%rax, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -44(%rbp)
	movq	$2, -40(%rbp)
	jmp	.L19
.L10:
	cmpl	$-1, -52(%rbp)
	je	.L20
	cmpl	$0, -52(%rbp)
	jne	.L21
	movq	$14, -40(%rbp)
	jmp	.L22
.L20:
	movq	$8, -40(%rbp)
	jmp	.L22
.L21:
	movq	$7, -40(%rbp)
	nop
.L22:
	jmp	.L19
.L16:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	$10, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtol@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	imulq	%rax, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -48(%rbp)
	movq	$10, -40(%rbp)
	jmp	.L19
.L13:
	movl	-68(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -48(%rbp)
	jg	.L23
	movq	$5, -40(%rbp)
	jmp	.L19
.L23:
	movq	$1, -40(%rbp)
	jmp	.L19
.L15:
	movl	-68(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	movq	$2, -40(%rbp)
	jmp	.L19
.L17:
	movl	-44(%rbp), %eax
	cmpl	-68(%rbp), %eax
	jge	.L25
	movq	$19, -40(%rbp)
	jmp	.L19
.L25:
	movq	$13, -40(%rbp)
	jmp	.L19
.L28:
	nop
.L19:
	jmp	.L27
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
