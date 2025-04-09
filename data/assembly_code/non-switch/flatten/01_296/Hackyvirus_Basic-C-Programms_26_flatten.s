	.file	"Hackyvirus_Basic-C-Programms_26_flatten.c"
	.text
	.globl	_TIG_IZ_5b2r_argv
	.bss
	.align 8
	.type	_TIG_IZ_5b2r_argv, @object
	.size	_TIG_IZ_5b2r_argv, 8
_TIG_IZ_5b2r_argv:
	.zero	8
	.globl	_TIG_IZ_5b2r_argc
	.align 4
	.type	_TIG_IZ_5b2r_argc, @object
	.size	_TIG_IZ_5b2r_argc, 4
_TIG_IZ_5b2r_argc:
	.zero	4
	.globl	_TIG_IZ_5b2r_envp
	.align 8
	.type	_TIG_IZ_5b2r_envp, @object
	.size	_TIG_IZ_5b2r_envp, 8
_TIG_IZ_5b2r_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d + %d = %d\n"
	.align 8
.LC1:
	.string	"The sum of first 10 natural Numbers: %d\n"
.LC2:
	.string	"Using while loop"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_5b2r_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_5b2r_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_5b2r_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 91 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5b2r--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_5b2r_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_5b2r_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_5b2r_envp(%rip)
	nop
	movq	$8, -8(%rbp)
.L23:
	cmpq	$15, -8(%rbp)
	ja	.L25
	movq	-8(%rbp), %rax
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
	.long	.L25-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L25-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-16(%rbp), %eax
	addl	%eax, -12(%rbp)
	movl	-12(%rbp), %ecx
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L17
.L9:
	cmpl	$10, -20(%rbp)
	jg	.L18
	movq	$2, -8(%rbp)
	jmp	.L17
.L18:
	movq	$11, -8(%rbp)
	jmp	.L17
.L7:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L17
.L12:
	movq	$7, -8(%rbp)
	jmp	.L17
.L15:
	cmpl	$10, -16(%rbp)
	jg	.L20
	movq	$4, -8(%rbp)
	jmp	.L17
.L20:
	movq	$15, -8(%rbp)
	jmp	.L17
.L10:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L17
.L11:
	movl	$0, %eax
	jmp	.L24
.L13:
	movl	$0, -24(%rbp)
	movl	$1, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L17
.L16:
	movl	-20(%rbp), %eax
	addl	%eax, -24(%rbp)
	movl	-24(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L17
.L25:
	nop
.L17:
	jmp	.L23
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
