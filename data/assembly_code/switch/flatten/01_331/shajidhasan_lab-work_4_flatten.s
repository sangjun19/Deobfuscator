	.file	"shajidhasan_lab-work_4_flatten.c"
	.text
	.globl	_TIG_IZ_gOjZ_argc
	.bss
	.align 4
	.type	_TIG_IZ_gOjZ_argc, @object
	.size	_TIG_IZ_gOjZ_argc, 4
_TIG_IZ_gOjZ_argc:
	.zero	4
	.globl	_TIG_IZ_gOjZ_argv
	.align 8
	.type	_TIG_IZ_gOjZ_argv, @object
	.size	_TIG_IZ_gOjZ_argv, 8
_TIG_IZ_gOjZ_argv:
	.zero	8
	.globl	_TIG_IZ_gOjZ_envp
	.align 8
	.type	_TIG_IZ_gOjZ_envp, @object
	.size	_TIG_IZ_gOjZ_envp, 8
_TIG_IZ_gOjZ_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The number is even."
.LC1:
	.string	"The number is odd."
.LC2:
	.string	"Enter a number: "
.LC3:
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_gOjZ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_gOjZ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_gOjZ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-gOjZ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_gOjZ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_gOjZ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_gOjZ_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L21:
	cmpq	$9, -16(%rbp)
	ja	.L24
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L24-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L24-.L8
	.long	.L7-.L8
	.text
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L16
.L14:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L16
.L7:
	movl	-20(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	testl	%eax, %eax
	je	.L17
	cmpl	$1, %eax
	jne	.L18
	movq	$1, -16(%rbp)
	jmp	.L19
.L17:
	movq	$4, -16(%rbp)
	jmp	.L19
.L18:
	movq	$6, -16(%rbp)
	nop
.L19:
	jmp	.L16
.L10:
	movq	$0, -16(%rbp)
	jmp	.L16
.L11:
	movq	$7, -16(%rbp)
	jmp	.L16
.L15:
	movl	$10, %edi
	call	putchar@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L16
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L22
	jmp	.L23
.L24:
	nop
.L16:
	jmp	.L21
.L23:
	call	__stack_chk_fail@PLT
.L22:
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
