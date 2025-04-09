	.file	"wztxy_BUAA_accoding_6_flatten.c"
	.text
	.globl	_TIG_IZ_t86z_envp
	.bss
	.align 8
	.type	_TIG_IZ_t86z_envp, @object
	.size	_TIG_IZ_t86z_envp, 8
_TIG_IZ_t86z_envp:
	.zero	8
	.globl	_TIG_IZ_t86z_argv
	.align 8
	.type	_TIG_IZ_t86z_argv, @object
	.size	_TIG_IZ_t86z_argv, 8
_TIG_IZ_t86z_argv:
	.zero	8
	.globl	_TIG_IZ_t86z_argc
	.align 4
	.type	_TIG_IZ_t86z_argc, @object
	.size	_TIG_IZ_t86z_argc, 4
_TIG_IZ_t86z_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d\n"
.LC1:
	.string	"%d%d%d\n"
.LC2:
	.string	"%d%d\n"
.LC3:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_t86z_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_t86z_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_t86z_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-t86z--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_t86z_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_t86z_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_t86z_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L26:
	cmpq	$11, -16(%rbp)
	ja	.L29
	movq	-16(%rbp), %rax
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L29-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L29-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movq	$0, -16(%rbp)
	jmp	.L18
.L10:
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L18
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	jmp	.L28
.L14:
	movl	-32(%rbp), %ecx
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L18
.L7:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1374389535, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$5, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -32(%rbp)
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	subl	%eax, %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %ecx
	movl	%ecx, %eax
	sall	$2, %eax
	addl	%ecx, %eax
	addl	%eax, %eax
	subl	%eax, %edx
	movl	%edx, -28(%rbp)
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %ecx
	movl	%ecx, %eax
	sall	$2, %eax
	addl	%ecx, %eax
	addl	%eax, %eax
	subl	%eax, %edx
	movl	%edx, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L18
.L12:
	cmpl	$0, -24(%rbp)
	je	.L20
	movq	$3, -16(%rbp)
	jmp	.L18
.L20:
	movq	$2, -16(%rbp)
	jmp	.L18
.L9:
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L18
.L17:
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -20(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L18
.L11:
	cmpl	$-1, -20(%rbp)
	je	.L22
	movq	$11, -16(%rbp)
	jmp	.L18
.L22:
	movq	$1, -16(%rbp)
	jmp	.L18
.L15:
	cmpl	$0, -28(%rbp)
	je	.L24
	movq	$10, -16(%rbp)
	jmp	.L18
.L24:
	movq	$8, -16(%rbp)
	jmp	.L18
.L29:
	nop
.L18:
	jmp	.L26
.L28:
	call	__stack_chk_fail@PLT
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
