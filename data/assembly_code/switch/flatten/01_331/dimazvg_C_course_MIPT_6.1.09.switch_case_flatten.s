	.file	"dimazvg_C_course_MIPT_6.1.09.switch_case_flatten.c"
	.text
	.globl	_TIG_IZ_HXMX_envp
	.bss
	.align 8
	.type	_TIG_IZ_HXMX_envp, @object
	.size	_TIG_IZ_HXMX_envp, 8
_TIG_IZ_HXMX_envp:
	.zero	8
	.globl	_TIG_IZ_HXMX_argc
	.align 4
	.type	_TIG_IZ_HXMX_argc, @object
	.size	_TIG_IZ_HXMX_argc, 4
_TIG_IZ_HXMX_argc:
	.zero	4
	.globl	_TIG_IZ_HXMX_argv
	.align 8
	.type	_TIG_IZ_HXMX_argv, @object
	.size	_TIG_IZ_HXMX_argv, 8
_TIG_IZ_HXMX_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d \320\272\320\276\321\200\320\276\320\262\321\213"
.LC2:
	.string	"%d \320\272\320\276\321\200\320\276\320\262"
.LC3:
	.string	"%d \320\272\320\276\321\200\320\276\320\262\320\260"
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
	movq	$0, _TIG_IZ_HXMX_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_HXMX_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_HXMX_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HXMX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HXMX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HXMX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HXMX_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L19:
	cmpq	$7, -16(%rbp)
	ja	.L22
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L22-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L12:
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -16(%rbp)
	jmp	.L14
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	jmp	.L21
.L9:
	movl	-20(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %edx
	sarl	$31, %edx
	subl	%edx, %eax
	movl	%eax, %edx
	sall	$2, %edx
	addl	%eax, %edx
	leal	(%rdx,%rdx), %eax
	movl	%eax, %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	cmpl	$1, %eax
	je	.L16
	testl	%eax, %eax
	jle	.L17
	subl	$2, %eax
	cmpl	$2, %eax
	ja	.L17
	movq	$0, -16(%rbp)
	jmp	.L18
.L16:
	movq	$2, -16(%rbp)
	jmp	.L18
.L17:
	movq	$7, -16(%rbp)
	nop
.L18:
	jmp	.L14
.L13:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L7:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L11:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L22:
	nop
.L14:
	jmp	.L19
.L21:
	call	__stack_chk_fail@PLT
.L20:
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
