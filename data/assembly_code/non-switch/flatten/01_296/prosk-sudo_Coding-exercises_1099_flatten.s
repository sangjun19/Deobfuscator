	.file	"prosk-sudo_Coding-exercises_1099_flatten.c"
	.text
	.globl	_TIG_IZ_blwJ_envp
	.bss
	.align 8
	.type	_TIG_IZ_blwJ_envp, @object
	.size	_TIG_IZ_blwJ_envp, 8
_TIG_IZ_blwJ_envp:
	.zero	8
	.globl	_TIG_IZ_blwJ_argv
	.align 8
	.type	_TIG_IZ_blwJ_argv, @object
	.size	_TIG_IZ_blwJ_argv, 8
_TIG_IZ_blwJ_argv:
	.zero	8
	.globl	_TIG_IZ_blwJ_argc
	.align 4
	.type	_TIG_IZ_blwJ_argc, @object
	.size	_TIG_IZ_blwJ_argc, 4
_TIG_IZ_blwJ_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d "
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
	subq	$480, %rsp
	movl	%edi, -452(%rbp)
	movq	%rsi, -464(%rbp)
	movq	%rdx, -472(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_blwJ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_blwJ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_blwJ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-blwJ--0
# 0 "" 2
#NO_APP
	movl	-452(%rbp), %eax
	movl	%eax, _TIG_IZ_blwJ_argc(%rip)
	movq	-464(%rbp), %rax
	movq	%rax, _TIG_IZ_blwJ_argv(%rip)
	movq	-472(%rbp), %rax
	movq	%rax, _TIG_IZ_blwJ_envp(%rip)
	nop
	movq	$8, -424(%rbp)
.L51:
	cmpq	$43, -424(%rbp)
	ja	.L54
	movq	-424(%rbp), %rax
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
	.long	.L32-.L8
	.long	.L54-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L54-.L8
	.long	.L26-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L54-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L54-.L8
	.long	.L54-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L54-.L8
	.long	.L12-.L8
	.long	.L54-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L54-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L20:
	addl	$1, -432(%rbp)
	subl	$1, -428(%rbp)
	movq	$2, -424(%rbp)
	jmp	.L33
.L29:
	leaq	-416(%rbp), %rcx
	movl	-436(%rbp), %eax
	movslq	%eax, %rsi
	movl	-440(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -436(%rbp)
	movq	$0, -424(%rbp)
	jmp	.L33
.L16:
	movl	-428(%rbp), %eax
	movslq	%eax, %rcx
	movl	-432(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	$9, -416(%rbp,%rax,4)
	movq	$29, -424(%rbp)
	jmp	.L33
.L25:
	movl	-428(%rbp), %eax
	movslq	%eax, %rcx
	movl	-432(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	$9, -416(%rbp,%rax,4)
	addl	$1, -428(%rbp)
	movq	$2, -424(%rbp)
	jmp	.L33
.L27:
	movq	$27, -424(%rbp)
	jmp	.L33
.L30:
	movl	$0, -436(%rbp)
	movq	$19, -424(%rbp)
	jmp	.L33
.L24:
	cmpl	$9, -440(%rbp)
	jg	.L34
	movq	$35, -424(%rbp)
	jmp	.L33
.L34:
	movq	$20, -424(%rbp)
	jmp	.L33
.L21:
	addl	$1, -440(%rbp)
	movq	$34, -424(%rbp)
	jmp	.L33
.L19:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L52
	jmp	.L53
.L23:
	cmpl	$9, -436(%rbp)
	jg	.L37
	movq	$10, -424(%rbp)
	jmp	.L33
.L37:
	movq	$42, -424(%rbp)
	jmp	.L33
.L10:
	cmpl	$9, -440(%rbp)
	jg	.L39
	movq	$3, -424(%rbp)
	jmp	.L33
.L39:
	movq	$26, -424(%rbp)
	jmp	.L33
.L18:
	movl	$1, -432(%rbp)
	movl	$1, -428(%rbp)
	movl	$0, -440(%rbp)
	movq	$16, -424(%rbp)
	jmp	.L33
.L14:
	cmpl	$9, -440(%rbp)
	jg	.L41
	movq	$7, -424(%rbp)
	jmp	.L33
.L41:
	movq	$2, -424(%rbp)
	jmp	.L33
.L15:
	cmpl	$9, -436(%rbp)
	jg	.L43
	movq	$43, -424(%rbp)
	jmp	.L33
.L43:
	movq	$39, -424(%rbp)
	jmp	.L33
.L12:
	movl	-428(%rbp), %eax
	movslq	%eax, %rcx
	movl	-432(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	-416(%rbp,%rax,4), %eax
	cmpl	$1, %eax
	jne	.L45
	movq	$25, -424(%rbp)
	jmp	.L33
.L45:
	movq	$30, -424(%rbp)
	jmp	.L33
.L26:
	movl	-436(%rbp), %eax
	movslq	%eax, %rcx
	movl	-440(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	-416(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -436(%rbp)
	movq	$19, -424(%rbp)
	jmp	.L33
.L9:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -440(%rbp)
	movq	$40, -424(%rbp)
	jmp	.L33
.L32:
	cmpl	$9, -436(%rbp)
	jg	.L47
	movq	$4, -424(%rbp)
	jmp	.L33
.L47:
	movq	$21, -424(%rbp)
	jmp	.L33
.L11:
	addl	$1, -440(%rbp)
	movq	$16, -424(%rbp)
	jmp	.L33
.L28:
	movl	$0, -436(%rbp)
	movq	$0, -424(%rbp)
	jmp	.L33
.L13:
	movl	$0, -436(%rbp)
	movq	$33, -424(%rbp)
	jmp	.L33
.L17:
	movl	$0, -440(%rbp)
	movq	$40, -424(%rbp)
	jmp	.L33
.L7:
	movl	-436(%rbp), %eax
	movslq	%eax, %rcx
	movl	-440(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	$0, -416(%rbp,%rax,4)
	addl	$1, -436(%rbp)
	movq	$33, -424(%rbp)
	jmp	.L33
.L31:
	movl	-428(%rbp), %eax
	movslq	%eax, %rcx
	movl	-432(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	movl	-416(%rbp,%rax,4), %eax
	testl	%eax, %eax
	jne	.L49
	movq	$15, -424(%rbp)
	jmp	.L33
.L49:
	movq	$37, -424(%rbp)
	jmp	.L33
.L22:
	movl	$0, -440(%rbp)
	movq	$34, -424(%rbp)
	jmp	.L33
.L54:
	nop
.L33:
	jmp	.L51
.L53:
	call	__stack_chk_fail@PLT
.L52:
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
