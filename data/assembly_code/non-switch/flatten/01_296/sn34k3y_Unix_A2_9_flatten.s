	.file	"sn34k3y_Unix_A2_9_flatten.c"
	.text
	.globl	_TIG_IZ_u07Y_argc
	.bss
	.align 4
	.type	_TIG_IZ_u07Y_argc, @object
	.size	_TIG_IZ_u07Y_argc, 4
_TIG_IZ_u07Y_argc:
	.zero	4
	.globl	_TIG_IZ_u07Y_envp
	.align 8
	.type	_TIG_IZ_u07Y_envp, @object
	.size	_TIG_IZ_u07Y_envp, 8
_TIG_IZ_u07Y_envp:
	.zero	8
	.globl	_TIG_IZ_u07Y_argv
	.align 8
	.type	_TIG_IZ_u07Y_argv, @object
	.size	_TIG_IZ_u07Y_argv, 8
_TIG_IZ_u07Y_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"fork"
.LC1:
	.string	"Child received: %s\n"
.LC2:
	.string	"pipe"
.LC3:
	.string	"hello world"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
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
	movq	$0, _TIG_IZ_u07Y_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_u07Y_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_u07Y_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-u07Y--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_u07Y_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_u07Y_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_u07Y_envp(%rip)
	nop
	movq	$8, -56(%rbp)
.L29:
	cmpq	$14, -56(%rbp)
	ja	.L32
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
	.long	.L20-.L8
	.long	.L32-.L8
	.long	.L19-.L8
	.long	.L32-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L18:
	movl	-40(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	leaq	1(%rax), %rdx
	movl	-36(%rbp), %eax
	movq	-64(%rbp), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-36(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$14, -56(%rbp)
	jmp	.L21
.L7:
	movl	$0, %eax
	jmp	.L30
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$10, -56(%rbp)
	jmp	.L21
.L14:
	movq	$7, -56(%rbp)
	jmp	.L21
.L11:
	cmpl	$-1, -68(%rbp)
	jne	.L23
	movq	$13, -56(%rbp)
	jmp	.L21
.L23:
	movq	$5, -56(%rbp)
	jmp	.L21
.L13:
	movl	-36(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-40(%rbp), %eax
	leaq	-32(%rbp), %rcx
	movl	$20, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-40(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$14, -56(%rbp)
	jmp	.L21
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$6, -56(%rbp)
	jmp	.L21
.L16:
	movl	$1, %eax
	jmp	.L30
.L17:
	call	fork@PLT
	movl	%eax, -72(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L21
.L12:
	movl	$1, %eax
	jmp	.L30
.L20:
	cmpl	$0, -72(%rbp)
	jne	.L25
	movq	$9, -56(%rbp)
	jmp	.L21
.L25:
	movq	$4, -56(%rbp)
	jmp	.L21
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, -64(%rbp)
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	pipe@PLT
	movl	%eax, -68(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L21
.L19:
	cmpl	$0, -72(%rbp)
	jns	.L27
	movq	$12, -56(%rbp)
	jmp	.L21
.L27:
	movq	$0, -56(%rbp)
	jmp	.L21
.L32:
	nop
.L21:
	jmp	.L29
.L30:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
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
