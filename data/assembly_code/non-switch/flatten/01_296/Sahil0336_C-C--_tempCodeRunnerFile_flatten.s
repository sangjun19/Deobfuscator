	.file	"Sahil0336_C-C--_tempCodeRunnerFile_flatten.c"
	.text
	.globl	_TIG_IZ_Acdl_envp
	.bss
	.align 8
	.type	_TIG_IZ_Acdl_envp, @object
	.size	_TIG_IZ_Acdl_envp, 8
_TIG_IZ_Acdl_envp:
	.zero	8
	.globl	_TIG_IZ_Acdl_argc
	.align 4
	.type	_TIG_IZ_Acdl_argc, @object
	.size	_TIG_IZ_Acdl_argc, 4
_TIG_IZ_Acdl_argc:
	.zero	4
	.globl	_TIG_IZ_Acdl_argv
	.align 8
	.type	_TIG_IZ_Acdl_argv, @object
	.size	_TIG_IZ_Acdl_argv, 8
_TIG_IZ_Acdl_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Enter the number:  "
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
	movq	$0, _TIG_IZ_Acdl_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Acdl_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Acdl_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Acdl--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_Acdl_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_Acdl_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_Acdl_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L32:
	cmpq	$27, -16(%rbp)
	ja	.L35
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L16-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	addl	$1, -32(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L22
.L17:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L23
	movq	$14, -16(%rbp)
	jmp	.L22
.L23:
	movq	$20, -16(%rbp)
	jmp	.L22
.L13:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -24(%rbp)
	addl	$1, -32(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L22
.L20:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jg	.L25
	movq	$25, -16(%rbp)
	jmp	.L22
.L25:
	movq	$13, -16(%rbp)
	jmp	.L22
.L18:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -32(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L22
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L15:
	subl	$1, -20(%rbp)
	movl	$1, -32(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L22
.L14:
	subl	$1, -24(%rbp)
	subl	$1, -24(%rbp)
	movl	$1, -32(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L22
.L7:
	movl	-28(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	$1, -32(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L22
.L10:
	movl	-32(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jg	.L28
	movq	$3, -16(%rbp)
	jmp	.L22
.L28:
	movq	$11, -16(%rbp)
	jmp	.L22
.L21:
	movq	$7, -16(%rbp)
	jmp	.L22
.L16:
	movl	$1, -24(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -20(%rbp)
	movl	$1, -28(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L22
.L19:
	movl	-36(%rbp), %eax
	cmpl	%eax, -28(%rbp)
	jg	.L30
	movq	$27, -16(%rbp)
	jmp	.L22
.L30:
	movq	$21, -16(%rbp)
	jmp	.L22
.L12:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -28(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L22
.L35:
	nop
.L22:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
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
