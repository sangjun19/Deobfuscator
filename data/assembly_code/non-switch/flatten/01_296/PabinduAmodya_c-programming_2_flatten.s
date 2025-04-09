	.file	"PabinduAmodya_c-programming_2_flatten.c"
	.text
	.globl	_TIG_IZ_oTob_argc
	.bss
	.align 4
	.type	_TIG_IZ_oTob_argc, @object
	.size	_TIG_IZ_oTob_argc, 4
_TIG_IZ_oTob_argc:
	.zero	4
	.globl	_TIG_IZ_oTob_argv
	.align 8
	.type	_TIG_IZ_oTob_argv, @object
	.size	_TIG_IZ_oTob_argv, 8
_TIG_IZ_oTob_argv:
	.zero	8
	.globl	_TIG_IZ_oTob_envp
	.align 8
	.type	_TIG_IZ_oTob_envp, @object
	.size	_TIG_IZ_oTob_envp, 8
_TIG_IZ_oTob_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"The minimum value in the array is: %d\n"
	.align 8
.LC1:
	.string	"Enter the value of element %d: "
.LC2:
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_oTob_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_oTob_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_oTob_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oTob--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_oTob_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_oTob_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_oTob_envp(%rip)
	nop
	movq	$10, -56(%rbp)
.L26:
	cmpq	$12, -56(%rbp)
	ja	.L29
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L29-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L29-.L8
	.long	.L11-.L8
	.long	.L29-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	cmpl	$9, -68(%rbp)
	jg	.L18
	movq	$2, -56(%rbp)
	jmp	.L20
.L18:
	movq	$11, -56(%rbp)
	jmp	.L20
.L7:
	movl	-60(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	cmpl	%eax, -64(%rbp)
	jle	.L21
	movq	$5, -56(%rbp)
	jmp	.L20
.L21:
	movq	$0, -56(%rbp)
	jmp	.L20
.L11:
	cmpl	$9, -60(%rbp)
	jg	.L23
	movq	$12, -56(%rbp)
	jmp	.L20
.L23:
	movq	$1, -56(%rbp)
	jmp	.L20
.L16:
	movl	-64(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -56(%rbp)
	jmp	.L20
.L9:
	movl	-48(%rbp), %eax
	movl	%eax, -64(%rbp)
	movl	$1, -60(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L20
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	jmp	.L28
.L13:
	movl	-60(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	movl	%eax, -64(%rbp)
	movq	$0, -56(%rbp)
	jmp	.L20
.L10:
	movl	$0, -68(%rbp)
	movq	$4, -56(%rbp)
	jmp	.L20
.L17:
	addl	$1, -60(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L20
.L15:
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rdx
	movl	-68(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -68(%rbp)
	movq	$4, -56(%rbp)
	jmp	.L20
.L29:
	nop
.L20:
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
