	.file	"shreyansh-sawarn_Data-Structures-And-Algorithms_is_flatten.c"
	.text
	.globl	_TIG_IZ_QhBe_argv
	.bss
	.align 8
	.type	_TIG_IZ_QhBe_argv, @object
	.size	_TIG_IZ_QhBe_argv, 8
_TIG_IZ_QhBe_argv:
	.zero	8
	.globl	_TIG_IZ_QhBe_argc
	.align 4
	.type	_TIG_IZ_QhBe_argc, @object
	.size	_TIG_IZ_QhBe_argc, 4
_TIG_IZ_QhBe_argc:
	.zero	4
	.globl	_TIG_IZ_QhBe_envp
	.align 8
	.type	_TIG_IZ_QhBe_envp, @object
	.size	_TIG_IZ_QhBe_envp, 8
_TIG_IZ_QhBe_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
	.align 8
.LC1:
	.string	"Enter the number of elements to be sorted: "
.LC2:
	.string	"Enter %d elements: "
.LC3:
	.string	"Sorted array: "
.LC4:
	.string	" %d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_QhBe_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_QhBe_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_QhBe_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-QhBe--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_QhBe_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_QhBe_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_QhBe_envp(%rip)
	nop
	movq	$29, -120(%rbp)
.L36:
	cmpq	$29, -120(%rbp)
	ja	.L39
	movq	-120(%rbp), %rax
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
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L20-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L39-.L8
	.long	.L13-.L8
	.long	.L39-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L10-.L8
	.long	.L39-.L8
	.long	.L9-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L39-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	-132(%rbp), %eax
	cltq
	movl	-112(%rbp,%rax,4), %eax
	movl	%eax, -124(%rbp)
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -128(%rbp)
	movq	$11, -120(%rbp)
	jmp	.L24
.L15:
	movl	-128(%rbp), %eax
	leal	1(%rax), %ecx
	movl	-128(%rbp), %eax
	cltq
	movl	-112(%rbp,%rax,4), %edx
	movslq	%ecx, %rax
	movl	%edx, -112(%rbp,%rax,4)
	subl	$1, -128(%rbp)
	movq	$11, -120(%rbp)
	jmp	.L24
.L14:
	leaq	-112(%rbp), %rdx
	movl	-132(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -132(%rbp)
	movq	$23, -120(%rbp)
	jmp	.L24
.L17:
	movl	$10, %edi
	call	putchar@PLT
	movq	$0, -120(%rbp)
	jmp	.L24
.L22:
	movl	-136(%rbp), %eax
	cmpl	%eax, -132(%rbp)
	jge	.L25
	movq	$25, -120(%rbp)
	jmp	.L24
.L25:
	movq	$7, -120(%rbp)
	jmp	.L24
.L10:
	movl	-136(%rbp), %eax
	cmpl	%eax, -132(%rbp)
	jge	.L27
	movq	$15, -120(%rbp)
	jmp	.L24
.L27:
	movq	$10, -120(%rbp)
	jmp	.L24
.L18:
	movl	-128(%rbp), %eax
	cltq
	movl	-112(%rbp,%rax,4), %eax
	cmpl	%eax, -124(%rbp)
	jge	.L29
	movq	$19, -120(%rbp)
	jmp	.L24
.L29:
	movq	$20, -120(%rbp)
	jmp	.L24
.L16:
	movl	-136(%rbp), %eax
	cmpl	%eax, -132(%rbp)
	jge	.L31
	movq	$2, -120(%rbp)
	jmp	.L24
.L31:
	movq	$12, -120(%rbp)
	jmp	.L24
.L12:
	cmpl	$0, -128(%rbp)
	js	.L33
	movq	$14, -120(%rbp)
	jmp	.L24
.L33:
	movq	$20, -120(%rbp)
	jmp	.L24
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-136(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-136(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -132(%rbp)
	movq	$23, -120(%rbp)
	jmp	.L24
.L19:
	movl	$1, -132(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L24
.L23:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L37
	jmp	.L38
.L20:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -132(%rbp)
	movq	$13, -120(%rbp)
	jmp	.L24
.L7:
	movq	$17, -120(%rbp)
	jmp	.L24
.L21:
	movl	-132(%rbp), %eax
	cltq
	movl	-112(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -132(%rbp)
	movq	$13, -120(%rbp)
	jmp	.L24
.L11:
	movl	-128(%rbp), %eax
	addl	$1, %eax
	cltq
	movl	-124(%rbp), %edx
	movl	%edx, -112(%rbp,%rax,4)
	addl	$1, -132(%rbp)
	movq	$1, -120(%rbp)
	jmp	.L24
.L39:
	nop
.L24:
	jmp	.L36
.L38:
	call	__stack_chk_fail@PLT
.L37:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
